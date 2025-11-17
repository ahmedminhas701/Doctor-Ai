from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import json
import uuid
from pathlib import Path
from aboutModel import get_about_model_data
import tempfile
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Import your existing chatbot logic
from medical_ai import detect_and_clean_languages, get_language_code
from medical_ai import get_bot_instance, process_text_input , MyBot


from dotenv import load_dotenv
load_dotenv()
app = FastAPI(
    title="DoctorAI Chatbot API",
    description="Medical chatbot for disease prediction and health recommendations",
    version="1.0.0"
)
print("API_KEY from env:", os.getenv("API_KEY"))
@app.get("/check-key")
def check_key():
    return {"api_key": os.getenv("API_KEY")}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot instances storage
bot_sessions: Dict[str, MyBot] = {}

# Pydantic model for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: Optional[str] = "english"

class ChatResponse(BaseModel):
    success: bool
    response: str
    image_path: Optional[str] = None
    language: str
    session_id: str
    download_url: Optional[str] = None  # ✅ add this


class HealthReportRequest(BaseModel):
    name: str
    age: int
    gender: str
    location: Optional[str] = "N/A"
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

@app.on_event("startup")
async def startup_event():
    """Initialize required directories on startup"""
    os.makedirs("./static", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    print("DoctorAI API started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    bot_sessions.clear()
    print("DoctorAI API shutdown complete!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DoctorAI Chatbot API is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/chat/start", response_model=SessionResponse)
async def start_chat_session():
    """Start a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        bot_sessions[session_id] = get_bot_instance()
        
        return SessionResponse(
            session_id=session_id,
            message="New chat session started successfully!"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")
@app.post("/chat/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message to the chatbot"""
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            bot_sessions[session_id] = get_bot_instance()
        
        if session_id not in bot_sessions:
            bot_sessions[session_id] = get_bot_instance()
        
        bot_instance = bot_sessions[session_id]
        
        # Process the message
        result = process_text_input(request.message, bot_instance)

        # ✅ Extra logic: check if report generated
        download_url = None
        if result['success'] and "Report generated successfully" in result['response']:
            reports_dir = Path("./reports")
            report_files = sorted(reports_dir.glob("*_DoctorAI_Report.pdf"), key=os.path.getmtime)
            if report_files:
                report_filename = report_files[-1].name  # last generated file
                # ⚡ Absolute URL (replace with your actual server IP/domain)
                server_base_url = "http://127.0.0.1:8000"
                download_url = f"{server_base_url}/report/download/{report_filename}"

        return ChatResponse(
            success=result['success'],
            response=result['response'],
            image_path=os.path.basename(result['image_path']) if result['image_path'] else None,
            language=result['language'],
            session_id=session_id,
            download_url=download_url   # ✅ now absolute path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/chat/image/{image_name}")
async def get_image(image_name: str):
    """Serve generated images"""
    try:
        image_path = Path("./static") / image_name
        if image_path.exists():
            return FileResponse(
                path=str(image_path),
                media_type="image/png",
                filename=image_name
            )
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

@app.get("/report/download/{report_name}")
async def download_report(report_name: str):
    """Download generated PDF reports"""
    try:
        report_path = Path("./reports") / report_name
        if report_path.exists():
            return FileResponse(
                path=str(report_path),
                media_type="application/pdf",
                filename=report_name
            )
        else:
            raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving report: {str(e)}")

@app.delete("/chat/session/{session_id}")
async def end_chat_session(session_id: str):
    """End a chat session"""
    try:
        if session_id in bot_sessions:
            del bot_sessions[session_id]
            return {"message": f"Session {session_id} ended successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")

@app.get("/chat/sessions")
async def get_active_sessions():
    """Get list of active chat sessions"""
    try:
        return {
            "active_sessions": list(bot_sessions.keys()),
            "total_sessions": len(bot_sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sessions: {str(e)}")

@app.post("/health/symptoms")
async def analyze_symptoms(request: ChatRequest):
    """Dedicated endpoint for symptom analysis"""
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in bot_sessions:
            bot_sessions[session_id] = get_bot_instance()
        
        bot_instance = bot_sessions[session_id]
        
        # Process symptoms
        result = process_text_input(request.message, bot_instance)
        
        return ChatResponse(
            success=result['success'],
            response=result['response'],
            image_path=result['image_path'],
            language=result['language'],
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing symptoms: {str(e)}")

@app.get("/health/recommendations/{session_id}/{rec_type}")
async def get_recommendations(session_id: str, rec_type: str):
    """Get specific recommendations (diet/exercise/medicine)"""
    try:
        if session_id not in bot_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        bot_instance = bot_sessions[session_id]
        
        # Map recommendation types
        rec_map = {
            "diet": "diet recommendations",
            "exercise": "exercise recommendations", 
            "medicine": "medicine recommendations"
        }
        
        if rec_type not in rec_map:
            raise HTTPException(status_code=400, detail="Invalid recommendation type")
        
        # Process recommendation request
        result = process_text_input(rec_map[rec_type], bot_instance)
        
        return ChatResponse(
            success=result['success'],
            response=result['response'],
            image_path=result['image_path'],
            language=result['language'],
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/report/generate/{session_id}")
async def generate_health_report(session_id: str, report_request: HealthReportRequest):
    """Generate PDF health report"""
    try:
        if session_id not in bot_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        bot_instance = bot_sessions[session_id]
        
        # Simulate report generation request
        report_message = f"Generate my health report. My name is {report_request.name}, I am {report_request.age} years old, gender is {report_request.gender}, location is {report_request.location}"
        
        result = process_text_input(report_message, bot_instance)
        
        # Check if report was generated successfully
        if result['success'] and "Report generated successfully" in result['response']:
            # Find the generated report file
            reports_dir = Path("./reports")
            report_files = list(reports_dir.glob(f"{report_request.name.replace(' ', '_')}_DoctorAI_Report.pdf"))
            
            if report_files:
                report_filename = report_files[0].name
                return {
                    "success": True,
                    "message": "Report generated successfully",
                    "download_url": f"/report/download/{report_filename}",
                    "filename": report_filename
                }
        
        return {
            "success": False,
            "message": result['response']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/health/status")
async def health_check():
    """Detailed health check with system status"""
    try:
        return {
            "status": "healthy",
            "active_sessions": len(bot_sessions),
            "directories": {
                "static_exists": os.path.exists("./static"),
                "reports_exists": os.path.exists("./reports"),
                "model_exists": os.path.exists("./model"),
                "data_exists": os.path.exists("./data")
            },
            "api_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    


@app.get("/about")
def about_model():
    return get_about_model_data()
    
# -------------------- SPEECH TO TEXT (STT) - WHISPER --------------------
WHISPER_MODEL_SIZE = "small"
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

@app.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str | None = None
):
    """Convert speech to text using Whisper"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Convert audio to WAV PCM 16kHz mono
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        converted_path = tmp_path.replace(".wav", "_converted.wav")
        audio.export(converted_path, format="wav")

        # Transcribe with Whisper
        segments, info = whisper_model.transcribe(converted_path, language=language)
        result_text = " ".join([segment.text for segment in segments])

        # Cleanup
        os.remove(tmp_path)
        os.remove(converted_path)

        return {
            "text": result_text.strip(),
            "language": info.language
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)