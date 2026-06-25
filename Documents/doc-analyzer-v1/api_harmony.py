"""
Harmony UI API - Enhanced FastAPI with WebSocket Support
Provides REST endpoints and real-time WebSocket updates for the Harmony UI
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import json
import logging
import os
from pathlib import Path

from core.models import AnalysisConfig, AnalysisMode
from core.orchestrator import DocumentationAnalysisOrchestrator
from core.event_emitter import EventEmitter, EventType, AnalysisEvent

# S3 support (optional)
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Storage configuration
RESULTS_DIR = Path("./analysis_results")
S3_BUCKET = os.getenv("S3_RESULTS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# Initialize S3 client if configured
s3_client = None
if S3_AVAILABLE and S3_BUCKET:
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        logger.info(f"S3 storage enabled: {S3_BUCKET}")
    except Exception as e:
        logger.warning(f"Failed to initialize S3 client: {e}")
        s3_client = None

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Documentation Analyzer - Harmony UI API",
    description="Enhanced API with WebSocket support for real-time analysis tracking",
    version="2.0.0"
)

# CORS configuration for Harmony UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI (if directory exists)
STATIC_DIR = Path("./static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static UI files mounted at /static")
    
    # Serve index.html at root for SPA
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        """Serve the React UI"""
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"message": "UI not available, use /docs for API"})
    
    # Catch-all route for SPA routing (must be last)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def catch_all(full_path: str):
        """Catch-all for SPA routing"""
        # Don't intercept API routes
        if full_path.startswith(("api/", "ws/", "health", "docs", "openapi.json", "redoc")):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Serve index.html for all other routes (SPA)
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Not found")

# ============================================================================
# Data Models
# ============================================================================

class AnalysisStatus(str, Enum):
    """Analysis status enum"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PhaseStatus(str, Enum):
    """Phase status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StartAnalysisRequest(BaseModel):
    """Request to start new analysis"""
    services: List[str]  # Service names (URLs will be looked up or provided)
    mode: str = "quick"  # "quick" or "deep"
    max_pages: int = 50
    max_depth: int = 3
    force_playwright: bool = False
    debug_mode: bool = False


class AnalysisPhase(BaseModel):
    """Phase information"""
    name: str
    description: str
    status: PhaseStatus
    progress: float = 0.0
    startedAt: Optional[datetime] = None
    duration: int = 0  # seconds
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AnalysisStateResponse(BaseModel):
    """Complete analysis state"""
    executionId: str
    status: AnalysisStatus
    services: List[str]
    mode: str
    progress: float
    phases: List[AnalysisPhase]
    startedAt: datetime
    completedAt: Optional[datetime] = None
    error: Optional[Dict[str, Any]] = None
    canPause: bool = False
    canResume: bool = False
    canRetry: bool = False


class CheckpointInfo(BaseModel):
    """Checkpoint information"""
    checkpointId: str
    phaseName: str
    timestamp: datetime
    progress: float


class ErrorDetails(BaseModel):
    """Error information"""
    message: str
    phaseName: Optional[str] = None
    timestamp: datetime
    retryable: bool
    stack: Optional[str] = None


# ============================================================================
# In-Memory State Management
# ============================================================================

# Active analyses
analyses: Dict[str, Dict[str, Any]] = {}

# WebSocket connections
connections: Dict[str, List[WebSocket]] = {}

# Analysis control (pause/resume/cancel)
control_flags: Dict[str, Dict[str, bool]] = {}


def create_initial_phases(mode: str) -> List[Dict[str, Any]]:
    """Create initial phase definitions based on mode"""
    if mode == "quick":
        return [
            {
                "name": "crawling",
                "description": "Crawling documentation pages",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "analysis",
                "description": "Analyzing onboarding experience",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "reporting",
                "description": "Generating report",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            }
        ]
    else:  # deep mode
        return [
            {
                "name": "crawling",
                "description": "Crawling documentation",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "characterization",
                "description": "Service characterization & discovery",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "deep_analysis",
                "description": "Deep page analysis",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "resource_analysis",
                "description": "External resource analysis",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "journey_simulation",
                "description": "Persona journey simulation",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "findings",
                "description": "Generating findings",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            },
            {
                "name": "reporting",
                "description": "Generating report",
                "status": "pending",
                "progress": 0.0,
                "metadata": {}
            }
        ]


# ============================================================================
# WebSocket Event Broadcasting
# ============================================================================

async def broadcast_event(execution_id: str, event_type: str, data: Dict[str, Any]):
    """Broadcast event to all connected WebSocket clients for this analysis"""
    if execution_id not in connections:
        return
    
    message = {
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    dead_connections = []
    for websocket in connections[execution_id]:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to WebSocket: {e}")
            dead_connections.append(websocket)
    
    # Remove dead connections
    for ws in dead_connections:
        connections[execution_id].remove(ws)


async def send_status_update(execution_id: str):
    """Send complete status update to clients"""
    if execution_id not in analyses:
        return
    
    analysis = analyses[execution_id]
    await broadcast_event(execution_id, "status_update", {
        "status": analysis["status"],
        "progress": analysis["progress"],
        "currentPhase": analysis.get("currentPhase")
    })


async def send_phase_update(execution_id: str, phase_name: str, updates: Dict[str, Any]):
    """Send phase-specific update"""
    await broadcast_event(execution_id, "phase_update", {
        "phaseName": phase_name,
        **updates
    })


async def send_error(execution_id: str, error: Dict[str, Any]):
    """Send error notification"""
    await broadcast_event(execution_id, "error", error)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
def root():
    """API health check"""
    return {
        "service": "Documentation Analyzer - Harmony UI API",
        "version": "2.0.0",
        "status": "operational",
        "features": ["REST API", "WebSocket", "Real-time Updates"],
        "endpoints": {
            "POST /api/analysis/start": "Start new analysis",
            "GET /api/analysis/{execution_id}": "Get analysis status",
            "POST /api/analysis/{execution_id}/pause": "Pause analysis",
            "POST /api/analysis/{execution_id}/resume": "Resume analysis",
            "POST /api/analysis/{execution_id}/retry/{phase}": "Retry failed phase",
            "POST /api/analysis/{execution_id}/cancel": "Cancel analysis",
            "GET /api/analysis/{execution_id}/checkpoints": "Get checkpoints",
            "GET /api/analysis/{execution_id}/results": "Get results",
            "DELETE /api/analysis/{execution_id}": "Delete analysis",
            "WebSocket /ws/{execution_id}": "Real-time updates"
        }
    }


@app.post("/api/analysis/start")
async def start_analysis(
    request: StartAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start new documentation analysis
    
    Returns execution_id for tracking progress via WebSocket
    """
    execution_id = str(uuid.uuid4())
    
    # Initialize analysis state
    analyses[execution_id] = {
        "executionId": execution_id,
        "status": AnalysisStatus.STARTING,
        "services": request.services,
        "mode": request.mode,
        "progress": 0.0,
        "phases": create_initial_phases(request.mode),
        "startedAt": datetime.now(),
        "completedAt": None,
        "error": None,
        "config": request.dict(),
        "results": None,
        "checkpoints": []
    }
    
    # Initialize control flags
    control_flags[execution_id] = {
        "paused": False,
        "cancelled": False,
        "retry_phase": None
    }
    
    # Start analysis in background
    background_tasks.add_task(run_analysis_task, execution_id, request)
    
    return {
        "executionId": execution_id,
        "status": "starting",
        "message": "Analysis started successfully"
    }


@app.get("/api/analysis/{execution_id}")
@app.get("/api/analysis/{execution_id}/status")
def get_analysis_status(execution_id: str):
    """Get current analysis status"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[execution_id]
    
    return AnalysisStateResponse(
        executionId=analysis["executionId"],
        status=analysis["status"],
        services=analysis["services"],
        mode=analysis["mode"],
        progress=analysis["progress"],
        phases=[AnalysisPhase(**phase) for phase in analysis["phases"]],
        startedAt=analysis["startedAt"],
        completedAt=analysis["completedAt"],
        error=analysis["error"],
        canPause=analysis["status"] == AnalysisStatus.RUNNING,
        canResume=analysis["status"] == AnalysisStatus.PAUSED,
        canRetry=analysis["status"] == AnalysisStatus.FAILED
    )


@app.post("/api/analysis/{execution_id}/pause")
async def pause_analysis(execution_id: str):
    """Pause running analysis"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[execution_id]
    
    if analysis["status"] != AnalysisStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis is not running")
    
    control_flags[execution_id]["paused"] = True
    analysis["status"] = AnalysisStatus.PAUSED
    
    await send_status_update(execution_id)
    
    return {"message": "Analysis paused successfully"}


@app.post("/api/analysis/{execution_id}/resume")
async def resume_analysis(
    execution_id: str,
    checkpoint_id: Optional[str] = None
):
    """Resume paused analysis"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[execution_id]
    
    if analysis["status"] != AnalysisStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Analysis is not paused")
    
    control_flags[execution_id]["paused"] = False
    analysis["status"] = AnalysisStatus.RUNNING
    
    await send_status_update(execution_id)
    
    return {"message": "Analysis resumed successfully"}


@app.post("/api/analysis/{execution_id}/retry/{phase_name}")
async def retry_phase(execution_id: str, phase_name: str):
    """Retry a failed phase"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[execution_id]
    
    # Find the phase
    phase = next((p for p in analysis["phases"] if p["name"] == phase_name), None)
    if not phase:
        raise HTTPException(status_code=404, detail="Phase not found")
    
    if phase["status"] != PhaseStatus.FAILED:
        raise HTTPException(status_code=400, detail="Phase did not fail")
    
    # Reset phase
    phase["status"] = PhaseStatus.PENDING
    phase["error"] = None
    phase["progress"] = 0.0
    
    # Mark for retry
    control_flags[execution_id]["retry_phase"] = phase_name
    analysis["status"] = AnalysisStatus.RUNNING
    
    await send_phase_update(execution_id, phase_name, {
        "status": "pending",
        "message": "Phase queued for retry"
    })
    
    return {"message": f"Phase '{phase_name}' will be retried"}


@app.post("/api/analysis/{execution_id}/cancel")
async def cancel_analysis(execution_id: str):
    """Cancel running analysis"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[execution_id]
    
    if analysis["status"] in [AnalysisStatus.COMPLETED, AnalysisStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail="Analysis already finished")
    
    control_flags[execution_id]["cancelled"] = True
    analysis["status"] = AnalysisStatus.CANCELLED
    analysis["completedAt"] = datetime.now()
    
    await send_status_update(execution_id)
    
    return {"message": "Analysis cancelled successfully"}


@app.get("/api/analysis/{execution_id}/checkpoints")
def get_checkpoints(execution_id: str):
    """Get available checkpoints for analysis"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[execution_id]
    
    return {
        "executionId": execution_id,
        "checkpoints": [
            CheckpointInfo(**cp) for cp in analysis.get("checkpoints", [])
        ]
    }


@app.get("/api/analysis/{execution_id}/results")
def get_results(execution_id: str):
    """Get analysis results"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[execution_id]
    
    if analysis["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed (status: {analysis['status']})"
        )
    
    if not analysis.get("results"):
        raise HTTPException(status_code=404, detail="Results not available")
    
    return analysis["results"]


@app.delete("/api/analysis/{execution_id}")
def delete_analysis(execution_id: str):
    """Delete analysis data"""
    if execution_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Clean up
    if execution_id in analyses:
        del analyses[execution_id]
    if execution_id in control_flags:
        del control_flags[execution_id]
    if execution_id in connections:
        del connections[execution_id]
    
    return {"message": "Analysis deleted successfully"}


# ============================================================================
# File Download Endpoints
# ============================================================================

def find_analysis_dir(analysis_id: str) -> Optional[Path]:
    """Find analysis directory by ID (handles partial matches)"""
    # Check local filesystem
    if RESULTS_DIR.exists():
        for dir_path in RESULTS_DIR.iterdir():
            if dir_path.is_dir() and analysis_id in dir_path.name:
                return dir_path
    return None


def get_file_from_s3(analysis_id: str, filename: str) -> Optional[bytes]:
    """Download file from S3"""
    if not s3_client or not S3_BUCKET:
        return None
    
    try:
        # Find the actual folder name in S3
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"analysis_results/",
            Delimiter='/'
        )
        
        matching_folder = None
        for prefix in response.get('CommonPrefixes', []):
            folder = prefix['Prefix']
            if analysis_id in folder:
                matching_folder = folder
                break
        
        if not matching_folder:
            return None
        
        # Download file
        key = f"{matching_folder}{filename}"
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return obj['Body'].read()
        
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return None


@app.get("/api/analyses")
def list_analyses():
    """List all available analyses"""
    analyses_list = []
    
    # Check local filesystem
    if RESULTS_DIR.exists():
        for dir_path in RESULTS_DIR.iterdir():
            if dir_path.is_dir() and (dir_path / "analysis_results.json").exists():
                try:
                    with open(dir_path / "analysis_results.json") as f:
                        data = json.load(f)
                    analyses_list.append({
                        "id": dir_path.name,
                        "platforms": data.get("platforms", []),
                        "date": data.get("analysis_date"),
                        "findings_count": data.get("total_findings", 0)
                    })
                except Exception as e:
                    logger.error(f"Failed to read {dir_path}: {e}")
    
    # Check S3 if configured
    if s3_client and S3_BUCKET:
        try:
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix="analysis_results/",
                Delimiter='/'
            )
            for prefix in response.get('CommonPrefixes', []):
                folder = prefix['Prefix'].rstrip('/').split('/')[-1]
                # Check if already in local list
                if not any(a['id'] == folder for a in analyses_list):
                    analyses_list.append({
                        "id": folder,
                        "platforms": [],
                        "date": None,
                        "source": "s3"
                    })
        except Exception as e:
            logger.error(f"Failed to list S3 analyses: {e}")
    
    return {"analyses": analyses_list}


@app.get("/api/analyses/{analysis_id}/report")
def download_report(analysis_id: str):
    """Download main benchmarking report"""
    # Try local first
    analysis_dir = find_analysis_dir(analysis_id)
    if analysis_dir:
        report_path = analysis_dir / "BENCHMARKING_REPORT.md"
        if report_path.exists():
            return FileResponse(
                path=report_path,
                media_type="text/markdown",
                filename="BENCHMARKING_REPORT.md"
            )
    
    # Try S3
    content = get_file_from_s3(analysis_id, "BENCHMARKING_REPORT.md")
    if content:
        return JSONResponse(
            content={"filename": "BENCHMARKING_REPORT.md", "content": content.decode('utf-8')},
            media_type="application/json"
        )
    
    raise HTTPException(status_code=404, detail="Report not found")


@app.get("/api/analyses/{analysis_id}/appendix_b")
def download_appendix_b(analysis_id: str):
    """Download metrics appendix"""
    analysis_dir = find_analysis_dir(analysis_id)
    if analysis_dir:
        file_path = analysis_dir / "APPENDIX_B_METRICS.md"
        if file_path.exists():
            return FileResponse(
                path=file_path,
                media_type="text/markdown",
                filename="APPENDIX_B_METRICS.md"
            )
    
    content = get_file_from_s3(analysis_id, "APPENDIX_B_METRICS.md")
    if content:
        return JSONResponse(
            content={"filename": "APPENDIX_B_METRICS.md", "content": content.decode('utf-8')}
        )
    
    raise HTTPException(status_code=404, detail="Appendix B not found")


@app.get("/api/analyses/{analysis_id}/appendix_d")
def download_appendix_d(analysis_id: str):
    """Download journeys appendix"""
    analysis_dir = find_analysis_dir(analysis_id)
    if analysis_dir:
        file_path = analysis_dir / "APPENDIX_D_JOURNEYS.md"
        if file_path.exists():
            return FileResponse(
                path=file_path,
                media_type="text/markdown",
                filename="APPENDIX_D_JOURNEYS.md"
            )
    
    content = get_file_from_s3(analysis_id, "APPENDIX_D_JOURNEYS.md")
    if content:
        return JSONResponse(
            content={"filename": "APPENDIX_D_JOURNEYS.md", "content": content.decode('utf-8')}
        )
    
    raise HTTPException(status_code=404, detail="Appendix D not found")


@app.get("/api/analyses/{analysis_id}/appendix_f")
def download_appendix_f(analysis_id: str):
    """Download matrix appendix"""
    analysis_dir = find_analysis_dir(analysis_id)
    if analysis_dir:
        file_path = analysis_dir / "APPENDIX_F_MATRIX.md"
        if file_path.exists():
            return FileResponse(
                path=file_path,
                media_type="text/markdown",
                filename="APPENDIX_F_MATRIX.md"
            )
    
    content = get_file_from_s3(analysis_id, "APPENDIX_F_MATRIX.md")
    if content:
        return JSONResponse(
            content={"filename": "APPENDIX_F_MATRIX.md", "content": content.decode('utf-8')}
        )
    
    raise HTTPException(status_code=404, detail="Appendix F not found")


@app.get("/api/analyses/{analysis_id}/json")
def download_json(analysis_id: str):
    """Download raw JSON data"""
    analysis_dir = find_analysis_dir(analysis_id)
    if analysis_dir:
        file_path = analysis_dir / "analysis_results.json"
        if file_path.exists():
            return FileResponse(
                path=file_path,
                media_type="application/json",
                filename="analysis_results.json"
            )
    
    content = get_file_from_s3(analysis_id, "analysis_results.json")
    if content:
        return JSONResponse(content=json.loads(content.decode('utf-8')))
    
    raise HTTPException(status_code=404, detail="JSON data not found")


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Health check endpoint with storage info"""
    storage_info = {
        "type": "local",
        "path": str(RESULTS_DIR),
        "warning": "Ephemeral storage - files lost on restart"
    }
    
    if s3_client and S3_BUCKET:
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET)
            storage_info = {
                "type": "s3",
                "bucket": S3_BUCKET,
                "accessible": True
            }
        except Exception as e:
            storage_info = {
                "type": "s3_configured_but_inaccessible",
                "bucket": S3_BUCKET,
                "error": str(e)
            }
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "storage": storage_info
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/{execution_id}")
async def websocket_endpoint(websocket: WebSocket, execution_id: str):
    """
    WebSocket endpoint for real-time analysis updates
    
    Events sent:
    - status_update: Overall analysis status change
    - phase_update: Phase progress/status change
    - metadata_update: Phase metadata update (pages crawled, etc.)
    - error: Error occurred
    - checkpoint_saved: Checkpoint created
    - completed: Analysis completed
    - cancelled: Analysis cancelled
    """
    await websocket.accept()
    
    # Register connection
    if execution_id not in connections:
        connections[execution_id] = []
    connections[execution_id].append(websocket)
    
    logger.info(f"WebSocket connected for analysis: {execution_id}")
    
    # Send initial state
    if execution_id in analyses:
        try:
            await websocket.send_json({
                "type": "connected",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "executionId": execution_id,
                    "message": "Connected to analysis stream"
                }
            })
        except Exception as e:
            logger.error(f"Failed to send initial state: {e}")
    
    try:
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Handle any client commands if needed
            logger.debug(f"Received from client: {data}")
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for analysis: {execution_id}")
        if execution_id in connections:
            connections[execution_id].remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if execution_id in connections and websocket in connections[execution_id]:
            connections[execution_id].remove(websocket)


# ============================================================================
# Background Analysis Task
# ============================================================================

async def run_analysis_task(execution_id: str, request: StartAnalysisRequest):
    """
    Background task to run analysis with progress updates
    """
    try:
        analysis = analyses[execution_id]
        
        # Update to running
        analysis["status"] = AnalysisStatus.RUNNING
        await send_status_update(execution_id)
        
        # Simulate analysis phases (replace with actual orchestrator integration)
        phases = analysis["phases"]
        
        for i, phase in enumerate(phases):
            # Check control flags
            if control_flags[execution_id]["cancelled"]:
                logger.info(f"Analysis {execution_id} cancelled")
                return
            
            while control_flags[execution_id]["paused"]:
                await asyncio.sleep(1)
            
            # Start phase
            phase["status"] = PhaseStatus.RUNNING
            phase["startedAt"] = datetime.now()
            analysis["currentPhase"] = phase["name"]
            
            await send_phase_update(execution_id, phase["name"], {
                "status": "running",
                "message": f"Starting {phase['description']}"
            })
            
            # Simulate phase work with progress updates
            for progress in range(0, 101, 20):
                phase["progress"] = progress
                analysis["progress"] = (i + progress / 100) / len(phases) * 100
                
                await send_phase_update(execution_id, phase["name"], {
                    "progress": progress,
                    "metadata": {
                        "pagesCrawled": progress if phase["name"] == "crawling" else None
                    }
                })
                
                await asyncio.sleep(1)  # Simulate work
                
                # Check for pause/cancel
                if control_flags[execution_id]["cancelled"]:
                    return
                while control_flags[execution_id]["paused"]:
                    await asyncio.sleep(1)
            
            # Complete phase
            phase["status"] = PhaseStatus.COMPLETED
            phase["progress"] = 100
            phase["duration"] = 5  # seconds
            
            # Save checkpoint
            checkpoint = {
                "checkpointId": str(uuid.uuid4()),
                "phaseName": phase["name"],
                "timestamp": datetime.now(),
                "progress": analysis["progress"]
            }
            analysis["checkpoints"].append(checkpoint)
            
            await broadcast_event(execution_id, "checkpoint_saved", checkpoint)
        
        # Mark complete
        analysis["status"] = AnalysisStatus.COMPLETED
        analysis["completedAt"] = datetime.now()
        analysis["progress"] = 100.0
        
        # Mock results
        analysis["results"] = {
            "executionId": execution_id,
            "services": analysis["services"],
            "mode": analysis["mode"],
            "durationMinutes": 5,
            "completedAt": datetime.now().isoformat(),
            "reportMarkdown": "# Analysis Complete\n\nTest report generated.",
            "quickMetrics": {
                "onboardingScore": 8.5,
                "timeToHelloWorld": "15 minutes",
                "pagesAnalyzed": 50,
                "keyIssues": []
            } if request.mode == "quick" else None
        }
        
        await broadcast_event(execution_id, "completed", {
            "executionId": execution_id,
            "message": "Analysis completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        
        analysis["status"] = AnalysisStatus.FAILED
        analysis["completedAt"] = datetime.now()
        analysis["error"] = {
            "message": str(e),
            "phaseName": analysis.get("currentPhase"),
            "timestamp": datetime.now().isoformat(),
            "retryable": True
        }
        
        await send_error(execution_id, analysis["error"])


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("📡 Documentation Analyzer - Harmony UI API")
    print("="*80)
    print("🚀 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws/{execution_id}")
    print("🎨 Harmony UI: http://localhost:5173 (Vite dev server)")
    print("="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
