#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight HTTP load balancer for Reward Model services.
Supports round-robin and least-connections strategies.
"""
import asyncio
import aiohttp
from aiohttp import web
import logging
from collections import defaultdict
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
HEALTH_CHECK_INTERVAL = 10  # Health check interval in seconds
HEALTH_CHECK_TIMEOUT = 5    # Health check timeout in seconds
PROXY_TIMEOUT = 120         # Proxy request timeout in seconds
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB max body size

class LoadBalancer:
    def __init__(self, backends, strategy='least_conn'):
        """
        Initialize the load balancer.
        
        Args:
            backends: List of backend servers, format ['http://127.0.0.1:6001', ...]
            strategy: Load balancing strategy 'round_robin' or 'least_conn'
        """
        self.backends = backends
        self.strategy = strategy
        self.current_index = 0
        self.active_connections = defaultdict(int)
        self.backend_health = {backend: True for backend in backends}
        self.last_health_check = time.time()
        self.request_count = defaultdict(int)
        
        logger.info(f"LoadBalancer initialized with {len(backends)} backends")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Backends: {', '.join(backends)}")
        
    async def health_check(self):
        """Periodic health check for backends."""
        logger.info("Starting health check task...")
        while True:
            for backend in self.backends:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{backend}/health", 
                            timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
                        ) as resp:
                            was_healthy = self.backend_health[backend]
                            self.backend_health[backend] = (resp.status == 200 or resp.status == 404)
                            
                            if not was_healthy and self.backend_health[backend]:
                                logger.info(f"✓ Backend {backend} is now healthy")
                            elif was_healthy and not self.backend_health[backend]:
                                logger.warning(f"✗ Backend {backend} is now unhealthy (status: {resp.status})")
                                
                except Exception as e:
                    was_healthy = self.backend_health[backend]
                    self.backend_health[backend] = False
                    if was_healthy:
                        logger.error(f"✗ Backend {backend} is down: {e}")
            
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    
    def get_healthy_backends(self):
        """Get list of healthy backend servers."""
        return [b for b in self.backends if self.backend_health[b]]
    
    def select_backend(self):
        """Select backend server based on strategy."""
        healthy_backends = self.get_healthy_backends()
        
        if not healthy_backends:
            logger.error("No healthy backends available!")
            return None
        
        if self.strategy == 'round_robin':
            # Round-robin strategy
            backend = healthy_backends[self.current_index % len(healthy_backends)]
            self.current_index += 1
            return backend
        
        elif self.strategy == 'least_conn':
            # Least connections strategy
            backend = min(healthy_backends, key=lambda b: self.active_connections[b])
            return backend
        
        return healthy_backends[0]
    
    async def proxy_request(self, request):
        """Proxy request to backend server."""
        backend = self.select_backend()
        
        if not backend:
            return web.Response(text="No healthy backend available", status=503)
        
        # Track active connections
        self.active_connections[backend] += 1
        self.request_count[backend] += 1
        
        try:
            # Build backend URL
            path = request.path
            query = request.query_string
            url = f"{backend}{path}"
            if query:
                url += f"?{query}"
            
            # Prepare headers and data
            headers = dict(request.headers)
            headers.pop('Host', None)  # Remove Host header
            
            # Read request body
            data = await request.read()
            
            # Create session and send request
            timeout = aiohttp.ClientTimeout(total=PROXY_TIMEOUT)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    data=data,
                    allow_redirects=False
                ) as resp:
                    # Read response
                    body = await resp.read()
                    
                    # Build response
                    response = web.Response(
                        body=body,
                        status=resp.status,
                        headers=resp.headers
                    )
                    
                    logger.debug(f"✓ Proxied {request.method} {path} to {backend} - Status: {resp.status}")
                    return response
                    
        except asyncio.TimeoutError:
            logger.error(f"✗ Timeout when proxying to {backend}")
            return web.Response(text="Backend timeout", status=504)
        
        except Exception as e:
            logger.error(f"✗ Error proxying to {backend}: {e}")
            return web.Response(text=f"Backend error: {str(e)}", status=502)
        
        finally:
            # Decrease active connection count
            self.active_connections[backend] -= 1
    
    async def health_endpoint(self, request):
        """Health check endpoint."""
        healthy = len(self.get_healthy_backends()) > 0
        status = 200 if healthy else 503
        return web.Response(text="healthy\n" if healthy else "unhealthy\n", status=status)
    
    async def stats_endpoint(self, request):
        """Statistics endpoint."""
        healthy_backends = self.get_healthy_backends()
        stats = {
            "total_backends": len(self.backends),
            "healthy_backends": len(healthy_backends),
            "backend_status": {
                backend: {
                    "healthy": self.backend_health[backend],
                    "active_connections": self.active_connections[backend],
                    "total_requests": self.request_count[backend]
                }
                for backend in self.backends
            }
        }
        
        import json
        return web.Response(
            text=json.dumps(stats, indent=2),
            content_type='application/json'
        )

async def init_app():
    """Initialize application."""
    backends = [
        'http://127.0.0.1:6001',
        'http://127.0.0.1:6002',
        'http://127.0.0.1:6003',
        'http://127.0.0.1:6004',
    ]
    
    lb = LoadBalancer(backends, strategy='least_conn')
    
    # Start health check task
    asyncio.create_task(lb.health_check())
    
    app = web.Application(client_max_size=MAX_BODY_SIZE)
    
    # Routes
    app.router.add_route('GET', '/health', lb.health_endpoint)
    app.router.add_route('GET', '/stats', lb.stats_endpoint)
    app.router.add_route('*', '/{tail:.*}', lb.proxy_request)
    
    return app

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Reward Model Load Balancer")
    logger.info("=" * 60)
    logger.info("Listening on: http://0.0.0.0:8001")
    logger.info("Backend servers: 6001, 6002, 6003, 6004")
    logger.info("Strategy: Least Connections")
    logger.info("Health check endpoint: http://localhost:8001/health")
    logger.info("Stats endpoint: http://localhost:8001/stats")
    logger.info("=" * 60)
    
    web.run_app(
        init_app(),
        host='0.0.0.0',
        port=8001,
        access_log=logger
    )

