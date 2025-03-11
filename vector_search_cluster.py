"""
Web-Scale Vector Search System Architecture
------------------------------------------
This implementation provides the core architecture for scaling FAISS to hundreds of millions 
of vector searches with sub-millisecond latency.
"""

import os
import time
import uuid
import json
import asyncio
import logging
import threading
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
import redis
import zmq
import zmq.asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VectorSearchService")


class VectorCluster:
    """
    Manages a cluster of vector search nodes with distributed indexing and querying.
    """
    
    def __init__(
        self,
        vector_dim: int,
        num_shards: int = 4,
        replicas_per_shard: int = 2,
        cache_host: str = "localhost",
        cache_port: int = 6379,
        cache_ttl: int = 3600,  # 1 hour cache TTL
        cache_capacity: int = 100000
    ):
        """
        Initialize the vector cluster.
        
        Args:
            vector_dim: Dimensionality of vectors
            num_shards: Number of shards to divide the vector space
            replicas_per_shard: Number of replicas per shard for redundancy
            cache_host: Redis host for query cache
            cache_port: Redis port
            cache_ttl: Cache TTL in seconds
            cache_capacity: Maximum number of entries in cache
        """
        self.vector_dim = vector_dim
        self.num_shards = num_shards
        self.replicas_per_shard = replicas_per_shard
        
        # Node management
        self.nodes = {}  # node_id -> NodeInfo
        self.shard_to_nodes = {}  # shard_id -> [node_ids]
        self.ready_nodes = set()  # nodes ready to serve queries
        
        # Cache configuration
        self.cache_client = redis.Redis(host=cache_host, port=cache_port)
        self.cache_ttl = cache_ttl
        self.cache_capacity = cache_capacity
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "errors": 0,
            "latency_ms": []
        }
        
        # Query routing and load balancing
        self._init_routing()
        
        logger.info(f"Vector cluster initialized with {num_shards} shards, "
                   f"{replicas_per_shard} replicas per shard")
    
    def _init_routing(self):
        """Initialize query routing and load balancing."""
        self.node_load = {}  # node_id -> current load
        self.node_selector = ThreadLocalNodeSelector(self)
    
    def register_node(
        self, 
        node_id: str, 
        shard_id: int, 
        endpoint: str, 
        capacity: int
    ) -> bool:
        """
        Register a new node to the cluster.
        
        Args:
            node_id: Unique ID for the node
            shard_id: Shard ID this node belongs to
            endpoint: ZeroMQ endpoint for communication
            capacity: Maximum QPS this node can handle
            
        Returns:
            Success flag
        """
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already registered")
            return False
        
        # Register node
        self.nodes[node_id] = {
            "node_id": node_id,
            "shard_id": shard_id,
            "endpoint": endpoint,
            "capacity": capacity,
            "health": "initializing",
            "last_heartbeat": time.time()
        }
        
        # Update shard mapping
        if shard_id not in self.shard_to_nodes:
            self.shard_to_nodes[shard_id] = []
        self.shard_to_nodes[shard_id].append(node_id)
        
        # Initialize load tracking
        self.node_load[node_id] = 0
        
        logger.info(f"Registered node {node_id} for shard {shard_id}")
        return True
    
    def mark_node_ready(self, node_id: str) -> bool:
        """Mark a node as ready to serve queries."""
        if node_id not in self.nodes:
            return False
        
        self.nodes[node_id]["health"] = "healthy"
        self.ready_nodes.add(node_id)
        logger.info(f"Node {node_id} marked as ready")
        return True
    
    def heartbeat(self, node_id: str, status: Dict[str, Any]) -> bool:
        """Process heartbeat from a node with current status."""
        if node_id not in self.nodes:
            logger.warning(f"Heartbeat from unknown node {node_id}")
            return False
        
        self.nodes[node_id].update({
            "health": status.get("health", "unknown"),
            "last_heartbeat": time.time(),
            "metrics": status.get("metrics", {})
        })
        
        return True
    
    def get_nodes_for_shard(self, shard_id: int) -> List[str]:
        """Get all node IDs for a specific shard."""
        return self.shard_to_nodes.get(shard_id, [])
    
    def get_healthy_nodes_for_shard(self, shard_id: int) -> List[str]:
        """Get healthy node IDs for a specific shard."""
        all_nodes = self.get_nodes_for_shard(shard_id)
        return [node_id for node_id in all_nodes 
                if node_id in self.ready_nodes 
                and self.nodes[node_id]["health"] == "healthy"]
    
    def select_node_for_query(self, shard_id: int) -> Optional[str]:
        """
        Select the best node for a query using load balancing.
        
        Args:
            shard_id: Shard ID to query
            
        Returns:
            Selected node ID or None if no healthy nodes
        """
        return self.node_selector.select_node(shard_id)
    
    async def distributed_search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        search_all_shards: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a distributed search across shards.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            search_all_shards: Whether to search all shards or just primary
            
        Returns:
            Search results containing distances and IDs
        """
        # Generate a query ID
        query_id = str(uuid.uuid4())
        
        # Check cache first
        cache_key = f"query:{self._vector_to_cache_key(query_vector)}:{k}"
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.metrics["cache_hits"] += 1
            return cached_result
        
        start_time = time.time()
        
        # Determine which shards to query
        if search_all_shards:
            shards_to_query = list(range(self.num_shards))
        else:
            # If not searching all shards, use routing function to pick shard
            shard_id = self._route_query(query_vector)
            shards_to_query = [shard_id]
        
        # Collect results from all shards
        tasks = []
        for shard_id in shards_to_query:
            task = self._query_shard(shard_id, query_vector, k)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and merge results
        merged_results = self._merge_results(results, k)
        
        # Update metrics
        end_time = time.time()
        query_time_ms = (end_time - start_time) * 1000
        self.metrics["total_queries"] += 1
        self.metrics["latency_ms"].append(query_time_ms)
        # Keep only recent latency measurements
        if len(self.metrics["latency_ms"]) > 1000:
            self.metrics["latency_ms"] = self.metrics["latency_ms"][-1000:]
        
        # Cache the result
        self._cache_result(cache_key, merged_results)
        
        # Add query time to results
        merged_results["query_time_ms"] = query_time_ms
        merged_results["query_id"] = query_id
        
        return merged_results
    
    async def _query_shard(
        self,
        shard_id: int,
        query_vector: np.ndarray,
        k: int
    ) -> Dict[str, Any]:
        """Query a specific shard for results."""
        # Select a node for this shard
        node_id = self.select_node_for_query(shard_id)
        if not node_id:
            logger.warning(f"No healthy nodes for shard {shard_id}")
            return {"error": f"No healthy nodes for shard {shard_id}"}
        
        # Update load counter
        self.node_load[node_id] += 1
        
        try:
            # Get node endpoint
            endpoint = self.nodes[node_id]["endpoint"]
            
            # Send query to node
            context = zmq.asyncio.Context.instance()
            socket = context.socket(zmq.REQ)
            socket.connect(endpoint)
            
            # Prepare query
            query_data = {
                "vector": query_vector.tolist(),
                "k": k
            }
            
            # Send and receive
            await socket.send_json(query_data)
            response = await socket.recv_json()
            
            # Add shard info to response
            response["shard_id"] = shard_id
            response["node_id"] = node_id
            
            return response
        except Exception as e:
            logger.error(f"Error querying node {node_id}: {str(e)}")
            self.metrics["errors"] += 1
            return {"error": str(e), "shard_id": shard_id}
        finally:
            # Update load counter
            self.node_load[node_id] -= 1
    
    def _merge_results(
        self,
        shard_results: List[Dict[str, Any]],
        k: int
    ) -> Dict[str, Any]:
        """
        Merge results from multiple shards.
        
        Args:
            shard_results: Results from each shard
            k: Number of top results to keep
            
        Returns:
            Merged results
        """
        # Check for errors
        errors = [r for r in shard_results if "error" in r]
        if len(errors) == len(shard_results):
            # All shards failed
            return {"error": "All shards failed to respond", "details": errors}
        
        # Collect valid results
        all_distances = []
        all_indices = []
        all_metadata = []
        
        for result in shard_results:
            if "error" not in result:
                distances = result.get("distances", [])
                indices = result.get("indices", [])
                metadata = result.get("metadata", [])
                
                # Add shard_id to indices to make them globally unique
                shard_id = result.get("shard_id", 0)
                global_indices = [(shard_id, idx) for idx in indices]
                
                all_distances.extend(distances)
                all_indices.extend(global_indices)
                all_metadata.extend(metadata if metadata else [None] * len(indices))
        
        # Sort by distance
        combined = list(zip(all_distances, all_indices, all_metadata))
        combined.sort(key=lambda x: x[0])
        
        # Take top k
        top_k = combined[:k]
        
        # Unzip results
        if top_k:
            distances, indices, metadata = zip(*top_k)
        else:
            distances, indices, metadata = [], [], []
        
        return {
            "distances": list(distances),
            "indices": list(indices),
            "metadata": list(metadata) if any(metadata) else None,
            "num_shards_queried": len(shard_results) - len(errors),
            "num_shards_failed": len(errors)
        }
    
    def _route_query(self, query_vector: np.ndarray) -> int:
        """
        Route a query to a specific shard based on vector content.
        
        This is a simple routing function that can be improved with
        more advanced techniques like locality-sensitive hashing.
        
        Args:
            query_vector: Query vector
            
        Returns:
            Shard ID to route to
        """
        # Simple hash-based routing
        vector_sum = np.sum(query_vector)
        return int(abs(hash(vector_sum)) % self.num_shards)
    
    def _vector_to_cache_key(self, vector: np.ndarray) -> str:
        """Convert vector to cache key string."""
        # Use first 8 values and last 8 values to form a compact fingerprint
        if len(vector) > 16:
            fingerprint = np.concatenate([vector[:8], vector[-8:]])
        else:
            fingerprint = vector
        
        # Convert to string with limited precision
        return ",".join(f"{x:.5f}" for x in fingerprint)
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if a query result is in cache."""
        try:
            cached = self.cache_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache lookup error: {str(e)}")
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> bool:
        """Cache a query result."""
        try:
            # Don't cache errors
            if "error" in result:
                return False
            
            # Serialize and cache
            serialized = json.dumps(result)
            self.cache_client.setex(cache_key, self.cache_ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache storage error: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.metrics.copy()
        
        # Calculate latency percentiles
        if metrics["latency_ms"]:
            latencies = sorted(metrics["latency_ms"])
            n = len(latencies)
            metrics["p50_latency_ms"] = latencies[n // 2]
            metrics["p95_latency_ms"] = latencies[int(n * 0.95)]
            metrics["p99_latency_ms"] = latencies[int(n * 0.99)]
            metrics["avg_latency_ms"] = sum(latencies) / n
        
        # Add cache hit rate
        if metrics["total_queries"] > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["total_queries"]
        
        # Add node stats
        metrics["total_nodes"] = len(self.nodes)
        metrics["healthy_nodes"] = sum(1 for nid, info in self.nodes.items() 
                                     if info["health"] == "healthy")
        
        return metrics


class ThreadLocalNodeSelector:
    """Thread-local node selector for load balancing."""
    
    def __init__(self, cluster):
        self.cluster = cluster
        self.local = threading.local()
    
    def select_node(self, shard_id: int) -> Optional[str]:
        """Select a node for a shard using round-robin + health checks."""
        # Get healthy nodes for this shard
        healthy_nodes = self.cluster.get_healthy_nodes_for_shard(shard_id)
        if not healthy_nodes:
            return None
        
        # Initialize last index for this shard if needed
        if not hasattr(self.local, "last_index"):
            self.local.last_index = {}
        
        if shard_id not in self.local.last_index:
            self.local.last_index[shard_id] = -1
        
        # Select next node (round-robin)
        self.local.last_index[shard_id] = (self.local.last_index[shard_id] + 1) % len(healthy_nodes)
        return healthy_nodes[self.local.last_index[shard_id]]


class SearchNode:
    """
    A single node in the vector search cluster that manages a shard of the index.
    """
    
    def __init__(
        self,
        node_id: str,
        shard_id: int,
        vector_dim: int,
        index_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        bind_address: str = "tcp://0.0.0.0:0",  # 0 for auto-port
        coordinator_url: str = "http://localhost:8000"
    ):
        """
        Initialize a search node.
        
        Args:
            node_id: Unique ID for this node
            shard_id: Shard ID this node is responsible for
            vector_dim: Vector dimension
            index_path: Path to load index from
            gpu_id: GPU ID to use (None for CPU)
            bind_address: ZMQ binding address
            coordinator_url: URL to the coordinator service
        """
        self.node_id = node_id
        self.shard_id = shard_id
        self.vector_dim = vector_dim
        self.gpu_id = gpu_id
        self.coordinator_url = coordinator_url
        
        # Setup FAISS index
        self.index = None
        self.index_ready = False
        self.vector_count = 0
        
        # Setup ZMQ server
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port(bind_address)
        self.endpoint = f"{bind_address.split(':')[0]}:{bind_address.split(':')[1]}:{self.port}"
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "errors": 0,
            "latency_ms": []
        }
        
        # Load index if path provided
        if index_path:
            self.load_index(index_path)
        
        logger.info(f"Search node {node_id} initialized for shard {shard_id}")
        logger.info(f"Node endpoint: {self.endpoint}")
    
    def load_index(self, index_path: str) -> bool:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path: Path to the index file
            
        Returns:
            Success flag
        """
        try:
            logger.info(f"Loading index from {index_path}")
            
            # Load the CPU index
            cpu_index = faiss.read_index(index_path)
            
            # Move to GPU if specified
            if self.gpu_id is not None:
                logger.info(f"Moving index to GPU {self.gpu_id}")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
            else:
                self.index = cpu_index
            
            # Set index parameters for optimal performance
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(64, max(1, self.index.nlist // 8))
            
            if isinstance(self.index, faiss.IndexHNSW):
                self.index.hnsw.efSearch = 64
            
            # Update vector count
            self.vector_count = self.index.ntotal
            self.index_ready = True
            
            logger.info(f"Successfully loaded index with {self.vector_count} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Search the index for nearest neighbors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Search results containing distances and IDs
        """
        if not self.index_ready:
            return {"error": "Index not ready"}
        
        start_time = time.time()
        
        try:
            # Ensure proper shape and type
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            query_vector = query_vector.astype(np.float32)
            
            # Perform search
            distances, indices = self.index.search(query_vector, k)
            
            # Update metrics
            end_time = time.time()
            query_time_ms = (end_time - start_time) * 1000
            
            self.metrics["queries_processed"] += 1
            self.metrics["latency_ms"].append(query_time_ms)
            # Keep only recent latency measurements
            if len(self.metrics["latency_ms"]) > 1000:
                self.metrics["latency_ms"] = self.metrics["latency_ms"][-1000:]
            
            # Return results
            return {
                "distances": distances[0].tolist(),
                "indices": indices[0].tolist(),
                "query_time_ms": query_time_ms
            }
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            self.metrics["errors"] += 1
            return {"error": str(e)}
    
    def start(self) -> None:
        """Start the search node server."""
        logger.info(f"Starting search node {self.node_id}")
        
        # Register with coordinator
        self._register_with_coordinator()
        
        try:
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            
            # Start main query loop
            self._query_loop()
            
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully")
        except Exception as e:
            logger.error(f"Node error: {str(e)}")
        finally:
            self.socket.close()
            self.context.term()
    
    def _query_loop(self) -> None:
        """Main query processing loop."""
        logger.info("Starting query processing loop")
        
        while True:
            try:
                # Wait for query
                request = self.socket.recv_json()
                
                # Extract query parameters
                query_vector = np.array(request["vector"], dtype=np.float32)
                k = request.get("k", 10)
                
                # Process query
                result = self.search(query_vector, k)
                
                # Send response
                self.socket.send_json(result)
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                self.metrics["errors"] += 1
                
                # Send error response
                try:
                    self.socket.send_json({"error": str(e)})
                except:
                    pass
    
    def _register_with_coordinator(self) -> bool:
        """Register this node with the coordinator."""
        import requests
        
        try:
            # Calculate capacity based on hardware
            if self.gpu_id is not None:
                # GPU nodes have higher capacity
                capacity = 1000  # Queries per second
            else:
                capacity = 200  # Queries per second
            
            # Registration data
            reg_data = {
                "node_id": self.node_id,
                "shard_id": self.shard_id,
                "endpoint": self.endpoint,
                "capacity": capacity,
                "gpu_id": self.gpu_id
            }
            
            # Register with coordinator
            url = f"{self.coordinator_url}/api/v1/nodes/register"
            response = requests.post(url, json=reg_data)
            
            if response.status_code == 200:
                logger.info("Successfully registered with coordinator")
                
                # Mark as ready if index is loaded
                if self.index_ready:
                    ready_url = f"{self.coordinator_url}/api/v1/nodes/{self.node_id}/ready"
                    ready_response = requests.post(ready_url)
                    if ready_response.status_code == 200:
                        logger.info("Node marked as ready")
                    else:
                        logger.warning("Failed to mark node as ready")
                
                return True
            else:
                logger.error(f"Registration failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False
    
    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to coordinator."""
        import requests
        
        while True:
            try:
                # Calculate health status
                health = "healthy" if self.index_ready else "initializing"
                
                # Calculate latency percentiles
                latency_stats = {}
                if self.metrics["latency_ms"]:
                    latencies = sorted(self.metrics["latency_ms"])
                    n = len(latencies)
                    latency_stats["p50"] = latencies[n // 2]
                    latency_stats["p95"] = latencies[int(n * 0.95)]
                    latency_stats["p99"] = latencies[int(n * 0.99)]
                    latency_stats["avg"] = sum(latencies) / n
                
                # Prepare status data
                status = {
                    "health": health,
                    "vector_count": self.vector_count,
                    "metrics": {
                        "queries_processed": self.metrics["queries_processed"],
                        "errors": self.metrics["errors"],
                        "latency": latency_stats
                    }
                }
                
                # Send heartbeat
                url = f"{self.coordinator_url}/api/v1/nodes/{self.node_id}/heartbeat"
                response = requests.post(url, json=status)
                
                if response.status_code != 200:
                    logger.warning(f"Heartbeat failed: {response.text}")
                
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
            
            # Sleep before next heartbeat
            time.sleep(5)


# API models
class QueryRequest(BaseModel):
    vector: List[float]
    k: int = 10
    all_shards: bool = True


class NodeRegistration(BaseModel):
    node_id: str
    shard_id: int
    endpoint: str
    capacity: int
    gpu_id: Optional[int] = None


# FastAPI application
app = FastAPI(title="Vector Search Coordinator")
vector_cluster = None


@app.on_event("startup")
async def startup_event():
    """Initialize the vector cluster on startup."""
    global vector_cluster
    vector_cluster = VectorCluster(
        vector_dim=128,  # Default dimension, configurable
        num_shards=16,   # Default shard count, configurable
        replicas_per_shard=2,
        cache_host="localhost",
        cache_port=6379
    )
    logger.info("Vector search coordinator started")


@app.post("/api/v1/search")
async def search(query: QueryRequest):
    """
    Search for nearest neighbors of a query vector.
    """
    global vector_cluster
    
    query_vector = np.array(query.vector, dtype=np.float32)
    
    try:
        results = await vector_cluster.distributed_search(
            query_vector=query_vector,
            k=query.k,
            search_all_shards=query.all_shards
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/nodes/register")
async def register_node(registration: NodeRegistration):
    """
    Register a new search node with the cluster.
    """
    global vector_cluster
    
    success = vector_cluster.register_node(
        node_id=registration.node_id,
        shard_id=registration.shard_id,
        endpoint=registration.endpoint,
        capacity=registration.capacity
    )
    
    if success:
        return {"status": "registered"}
    else:
        raise HTTPException(status_code=400, detail="Registration failed")


@app.post("/api/v1/nodes/{node_id}/ready")
async def mark_node_ready(node_id: str):
    """
    Mark a node as ready to serve queries.
    """
    global vector_cluster
    
    success = vector_cluster.mark_node_ready(node_id)
    
    if success:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=404, detail="Node not found")


@app.post("/api/v1/nodes/{node_id}/heartbeat")
async def node_heartbeat(node_id: str, status: dict):
    """
    Process heartbeat from a node.
    """
    global vector_cluster
    
    success = vector_cluster.heartbeat(node_id, status)
    
    if success:
        return {"status": "acknowledged"}
    else:
        raise HTTPException(status_code=404, detail="Node not found")


@app.get("/api/v1/metrics")
async def get_metrics():
    """
    Get cluster performance metrics.
    """
    global vector_cluster
    
    return vector_cluster.get_metrics()


def start_coordinator(host: str = "0.0.0.0", port: int = 8000):
    """Start the coordinator service."""
    uvicorn.run(app, host=host, port=port)


def start_search_node(
    node_id: str,
    shard_id: int,
    vector_dim: int,
    index_path: str,
    gpu_id: Optional[int] = None,
    coordinator_url: str = "http://localhost:8000"
):
    """Start a search node."""
    node = SearchNode(
        node_id=node_id,
        shard_id=shard_id,
        vector_dim=vector_dim,
        index_path=index_path,
        gpu_id=gpu_id,
        coordinator_url=coordinator_url
    )
    
    # Start serving
    node.start()


def main():
    """Main entry point with command-line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-Scale Vector Search System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Coordinator command
    coord_parser = subparsers.add_parser("coordinator", help="Start the coordinator service")
    coord_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    coord_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    coord_parser.add_argument("--vector-dim", type=int, default=128, help="Vector dimension")
    coord_parser.add_argument("--num-shards", type=int, default=16, help="Number of shards")
    coord_parser.add_argument("--replicas", type=int, default=2, help="Replicas per shard")
    coord_parser.add_argument("--cache-host", type=str, default="localhost", help="Redis host")
    coord_parser.add_argument("--cache-port", type=int, default=6379, help="Redis port")
    
    # Node command
    node_parser = subparsers.add_parser("node", help="Start a search node")
    node_parser.add_argument("--node-id", type=str, required=True, help="Unique node ID")
    node_parser.add_argument("--shard-id", type=int, required=True, help="Shard ID")
    node_parser.add_argument("--vector-dim", type=int, default=128, help="Vector dimension")
    node_parser.add_argument("--index-path", type=str, required=True, help="Path to index file")
    node_parser.add_argument("--gpu-id", type=int, help="GPU ID to use (optional)")
    node_parser.add_argument("--coordinator", type=str, default="http://localhost:8000", 
                            help="Coordinator URL")
    
    # Index creation command
    index_parser = subparsers.add_parser("create-index", help="Create and shard an index")
    index_parser.add_argument("--vectors-path", type=str, required=True, help="Path to vectors file")
    index_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    index_parser.add_argument("--num-shards", type=int, default=16, help="Number of shards")
    index_parser.add_argument("--index-type", type=str, default="IVF", 
                             choices=["FLAT", "IVF", "IVFPQ", "HNSW"], help="Index type")
    
    args = parser.parse_args()
    
    if args.command == "coordinator":
        # Configure cluster parameters from arguments
        global vector_cluster
        app.state.vector_dim = args.vector_dim
        app.state.num_shards = args.num_shards
        app.state.replicas = args.replicas
        app.state.cache_host = args.cache_host
        app.state.cache_port = args.cache_port
        
        # Start coordinator
        logger.info(f"Starting coordinator on {args.host}:{args.port}")
        start_coordinator(host=args.host, port=args.port)
    
    elif args.command == "node":
        # Start search node
        logger.info(f"Starting search node {args.node_id} for shard {args.shard_id}")
        start_search_node(
            node_id=args.node_id,
            shard_id=args.shard_id,
            vector_dim=args.vector_dim,
            index_path=args.index_path,
            gpu_id=args.gpu_id,
            coordinator_url=args.coordinator
        )
    
    elif args.command == "create-index":
        # Create and shard an index
        create_sharded_index(
            vectors_path=args.vectors_path,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            index_type=args.index_type
        )
    
    else:
        parser.print_help()


def create_sharded_index(
    vectors_path: str,
    output_dir: str,
    num_shards: int = 16,
    index_type: str = "IVF"
):
    """
    Create a sharded FAISS index from a vectors file.
    
    Args:
        vectors_path: Path to vectors file (.npy)
        output_dir: Output directory for sharded indexes
        num_shards: Number of shards to create
        index_type: Type of index to create
    """
    logger.info(f"Creating {num_shards} sharded {index_type} indexes from {vectors_path}")
    
    # Load vectors
    vectors = np.load(vectors_path).astype(np.float32)
    vector_dim = vectors.shape[1]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle vectors for more even distribution
    np.random.shuffle(vectors)
    
    # Calculate vectors per shard
    vectors_per_shard = len(vectors) // num_shards
    remainder = len(vectors) % num_shards
    
    offset = 0
    for shard_id in range(num_shards):
        # Calculate shard size (distribute remainder evenly)
        shard_size = vectors_per_shard + (1 if shard_id < remainder else 0)
        
        # Get vectors for this shard
        shard_vectors = vectors[offset:offset+shard_size]
        offset += shard_size
        
        logger.info(f"Processing shard {shard_id} with {shard_size} vectors")
        
        # Create index for this shard
        if index_type == "FLAT":
            index = faiss.IndexFlatL2(vector_dim)
        
        elif index_type == "IVF":
            # For IVF, we need a quantizer
            quantizer = faiss.IndexFlatL2(vector_dim)
            # Number of clusters - rule of thumb: sqrt(num_vectors)
            n_clusters = max(100, int(np.sqrt(shard_size)))
            index = faiss.IndexIVFFlat(quantizer, vector_dim, n_clusters)
            # Need to train this index
            logger.info(f"Training IVF index for shard {shard_id}")
            index.train(shard_vectors)
        
        elif index_type == "IVFPQ":
            quantizer = faiss.IndexFlatL2(vector_dim)
            n_clusters = max(100, int(np.sqrt(shard_size)))
            # 8 bits per sub-vector, divide dimension by 4 (or at least 1)
            m = max(1, vector_dim // 4)
            bits = 8
            index = faiss.IndexIVFPQ(quantizer, vector_dim, n_clusters, m, bits)
            logger.info(f"Training IVFPQ index for shard {shard_id}")
            index.train(shard_vectors)
        
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(vector_dim, 32)  # 32 connections per node
            index.hnsw.efConstruction = 64  # More accurate construction
        
        # Add vectors to index
        logger.info(f"Adding {shard_size} vectors to shard {shard_id}")
        index.add(shard_vectors)
        
        # Save index
        output_path = os.path.join(output_dir, f"shard_{shard_id}.index")
        logger.info(f"Saving shard {shard_id} to {output_path}")
        faiss.write_index(index, output_path)
    
    # Save shard metadata
    metadata = {
        "num_shards": num_shards,
        "index_type": index_type,
        "vector_dim": vector_dim,
        "total_vectors": len(vectors),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "shards_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Successfully created {num_shards} sharded indexes in {output_dir}")


class DistributedVectorIndexBuilder:
    """
    Builds a distributed vector index from raw data.
    Supports processing large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        output_dir: str,
        vector_dim: int,
        num_shards: int = 16,
        index_type: str = "IVF",
        batch_size: int = 100000
    ):
        """
        Initialize the index builder.
        
        Args:
            output_dir: Output directory for sharded indexes
            vector_dim: Vector dimension
            num_shards: Number of shards to create
            index_type: Type of index to create
            batch_size: Batch size for processing
        """
        self.output_dir = output_dir
        self.vector_dim = vector_dim
        self.num_shards = num_shards
        self.index_type = index_type
        self.batch_size = batch_size
        
        # Create empty indexes
        self.indexes = self._create_empty_indexes()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_empty_indexes(self) -> List[faiss.Index]:
        """Create empty indexes for each shard."""
        indexes = []
        
        for shard_id in range(self.num_shards):
            if self.index_type == "FLAT":
                index = faiss.IndexFlatL2(self.vector_dim)
                indexes.append(index)
            
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.vector_dim)
                index = faiss.IndexIVFFlat(quantizer, self.vector_dim, 100)  # Initial clusters
                indexes.append(index)
            
            elif self.index_type == "IVFPQ":
                quantizer = faiss.IndexFlatL2(self.vector_dim)
                m = max(1, self.vector_dim // 4)
                index = faiss.IndexIVFPQ(quantizer, self.vector_dim, 100, m, 8)
                indexes.append(index)
            
            elif self.index_type == "HNSW":
                index = faiss.IndexHNSWFlat(self.vector_dim, 32)
                indexes.append(index)
        
        return indexes
    
    def _need_training(self) -> bool:
        """Check if the indexes need training."""
        if self.index_type in ["IVF", "IVFPQ"]:
            return True
        return False
    
    def process_file(self, vector_file: str) -> None:
        """
        Process a vector file and add to sharded indexes.
        
        Args:
            vector_file: Path to vector file (.npy)
        """
        logger.info(f"Processing vector file: {vector_file}")
        
        # Check if indexes need training
        needs_training = self._need_training()
        training_done = False
        
        # Process in batches to handle large files
        for batch_idx, vectors_batch in self._batch_iterator(vector_file):
            logger.info(f"Processing batch {batch_idx} with {len(vectors_batch)} vectors")
            
            # Train indexes if needed (only once with first batch)
            if needs_training and not training_done:
                self._train_indexes(vectors_batch)
                training_done = True
            
            # Shard and add vectors
            self._add_vectors_to_shards(vectors_batch)
    
    def _batch_iterator(self, vector_file: str):
        """Iterator that yields batches of vectors from a file."""
        # Map file to memory for efficient access
        mmap_mode = 'r'  # Read-only memory mapping
        
        try:
            # Memory-map the file
            vectors = np.load(vector_file, mmap_mode=mmap_mode)
            
            # Get total size
            total_vectors = vectors.shape[0]
            batch_size = min(self.batch_size, total_vectors)
            
            # Yield batches
            batch_idx = 0
            for start_idx in range(0, total_vectors, batch_size):
                end_idx = min(start_idx + batch_size, total_vectors)
                # Load batch into memory
                vectors_batch = vectors[start_idx:end_idx].astype(np.float32)
                yield batch_idx, vectors_batch
                batch_idx += 1
                
        except Exception as e:
            logger.error(f"Error processing file {vector_file}: {str(e)}")
            raise
    
    def _train_indexes(self, vectors: np.ndarray) -> None:
        """Train indexes that require training."""
        logger.info(f"Training {self.num_shards} indexes")
        
        for shard_id, index in enumerate(self.indexes):
            if hasattr(index, 'train'):
                logger.info(f"Training index for shard {shard_id}")
                index.train(vectors)
    
    def _add_vectors_to_shards(self, vectors: np.ndarray) -> None:
        """Add vectors to sharded indexes using hash-based sharding."""
        # Calculate a hash for each vector for distribution
        vector_hashes = np.sum(vectors, axis=1) % self.num_shards
        
        # Add vectors to appropriate shards
        for shard_id in range(self.num_shards):
            # Get vectors for this shard
            shard_mask = (vector_hashes == shard_id)
            shard_vectors = vectors[shard_mask]
            
            if len(shard_vectors) > 0:
                logger.info(f"Adding {len(shard_vectors)} vectors to shard {shard_id}")
                self.indexes[shard_id].add(shard_vectors)
    
    def save_indexes(self) -> None:
        """Save all sharded indexes to disk."""
        logger.info(f"Saving {self.num_shards} sharded indexes")
        
        total_vectors = 0
        
        for shard_id, index in enumerate(self.indexes):
            # Count vectors
            shard_vectors = index.ntotal
            total_vectors += shard_vectors
            
            # Save index
            output_path = os.path.join(self.output_dir, f"shard_{shard_id}.index")
            logger.info(f"Saving shard {shard_id} with {shard_vectors} vectors to {output_path}")
            faiss.write_index(index, output_path)
        
        # Save metadata
        metadata = {
            "num_shards": self.num_shards,
            "index_type": self.index_type,
            "vector_dim": self.vector_dim,
            "total_vectors": total_vectors,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(self.output_dir, "shards_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully saved {self.num_shards} sharded indexes with {total_vectors} total vectors")


if __name__ == "__main__":
    main()
