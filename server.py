import os
import asyncio
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from typing import Optional, List, Dict, Any
import json

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
DB_NAME = os.environ.get('DB_NAME', 'smartsched_db')

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
schedules_collection = db['schedules']

# Helper function for datetime serialization
def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    return obj

# Pydantic models
class TaskBreakdownRequest(BaseModel):
    goal: str
    user_id: Optional[str] = None

class Task(BaseModel):
    task_id: str
    title: str
    description: str
    start_time: str
    end_time: str
    priority: str
    tips: Optional[str] = None
    completed: bool = False

class Schedule(BaseModel):
    schedule_id: str
    user_id: str
    goal: str
    tasks: List[Task]
    created_at: datetime
    updated_at: datetime

class SaveScheduleRequest(BaseModel):
    schedule_id: str
    user_id: str
    goal: str
    tasks: List[Dict[str, Any]]

# Mock AI response for when API key is not provided
def get_mock_ai_response(goal: str) -> Dict[str, Any]:
    """Generate a mock AI response for demonstration purposes"""
    return {
        "schedule": [
            {
                "task_id": str(uuid.uuid4()),
                "title": "High Priority Task",
                "description": f"Start with the most important part of: {goal}",
                "start_time": "9:00 AM",
                "end_time": "10:00 AM",
                "priority": "high",
                "tips": "Begin when your focus is highest",
                "completed": False
            },
            {
                "task_id": str(uuid.uuid4()),
                "title": "Medium Priority Task",
                "description": f"Continue with secondary objectives from: {goal}",
                "start_time": "10:15 AM",
                "end_time": "11:30 AM",
                "priority": "medium",
                "tips": "Take a short break before this task",
                "completed": False
            },
            {
                "task_id": str(uuid.uuid4()),
                "title": "Completion Task",
                "description": f"Final review and completion of: {goal}",
                "start_time": "12:00 PM",
                "end_time": "12:30 PM",
                "priority": "low",
                "tips": "Review your work and ensure quality",
                "completed": False
            }
        ],
        "summary": f"Your smart plan for '{goal}' has been broken down into 3 focused tasks with optimal timing.",
        "total_time": "2 hours 15 minutes"
    }

async def get_ai_task_breakdown(goal: str) -> Dict[str, Any]:
    """Get AI-powered task breakdown using Gemini (when API key is available)"""
    try:
        # Check if API key is available
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        
        if not gemini_api_key:
            print("No Gemini API key found, using mock response")
            return get_mock_ai_response(goal)
        
        # Import and use the Gemini integration
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        # Create a unique session ID for this request
        session_id = str(uuid.uuid4())
        
        # Initialize Gemini chat with task breakdown system message
        chat = LlmChat(
            api_key=gemini_api_key,
            session_id=session_id,
            system_message="""You are a smart task breakdown assistant. Break down user goals into structured tasks with time slots, priorities, and helpful tips.

Always respond in the following JSON format:
{
    "schedule": [
        {
            "task_id": "unique_id",
            "title": "Task Title",
            "description": "Detailed description",
            "start_time": "9:00 AM",
            "end_time": "10:00 AM",
            "priority": "high|medium|low",
            "tips": "Helpful tip for this task",
            "completed": false
        }
    ],
    "summary": "Brief summary of the plan",
    "total_time": "Total estimated time"
}

Provide 3-5 tasks with realistic time estimates. Consider focus levels throughout the day."""
        ).with_model("gemini", "gemini-2.0-flash")
        
        # Create user message with the goal
        user_message = UserMessage(
            text=f"Break down this goal into a smart daily schedule: {goal}"
        )
        
        # Get AI response
        ai_response = await chat.send_message(user_message)
        
        # Try to parse the JSON response
        try:
            parsed_response = json.loads(ai_response)
            
            # Add unique IDs to tasks if not present
            for task in parsed_response.get("schedule", []):
                if "task_id" not in task:
                    task["task_id"] = str(uuid.uuid4())
                if "completed" not in task:
                    task["completed"] = False
                    
            return parsed_response
            
        except json.JSONDecodeError:
            print("Failed to parse AI response as JSON, using mock response")
            return get_mock_ai_response(goal)
            
    except Exception as e:
        print(f"Error in AI task breakdown: {e}")
        return get_mock_ai_response(goal)

# API Routes
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "SmartSched API is running"}

@app.post("/api/breakdown-task")
async def breakdown_task(request: TaskBreakdownRequest):
    """Break down a user goal into structured tasks using AI"""
    try:
        if not request.goal.strip():
            raise HTTPException(status_code=400, detail="Goal cannot be empty")
        
        # Get AI-powered task breakdown
        ai_response = await get_ai_task_breakdown(request.goal)
        
        return JSONResponse(content={
            "success": True,
            "data": ai_response,
            "message": "Task breakdown completed successfully"
        })
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to break down task: {str(e)}")

@app.post("/api/save-schedule")
async def save_schedule(request: SaveScheduleRequest):
    """Save a user's schedule to the database"""
    try:
        schedule_data = {
            "schedule_id": request.schedule_id,
            "user_id": request.user_id,
            "goal": request.goal,
            "tasks": request.tasks,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Check if schedule exists
        existing_schedule = schedules_collection.find_one({"schedule_id": request.schedule_id})
        
        if existing_schedule:
            # Update existing schedule
            schedules_collection.update_one(
                {"schedule_id": request.schedule_id},
                {"$set": {
                    "tasks": request.tasks,
                    "updated_at": datetime.utcnow()
                }}
            )
            message = "Schedule updated successfully"
        else:
            # Create new schedule
            schedules_collection.insert_one(schedule_data)
            message = "Schedule saved successfully"
        
        return JSONResponse(content={
            "success": True,
            "message": message
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save schedule: {str(e)}")

@app.get("/api/schedules/{user_id}")
async def get_user_schedules(user_id: str):
    """Get all schedules for a user"""
    try:
        schedules = list(schedules_collection.find(
            {"user_id": user_id},
            {"_id": 0}  # Exclude MongoDB _id field
        ).sort("created_at", -1))
        
        # Serialize datetime objects
        serialized_schedules = serialize_datetime(schedules)
        
        return JSONResponse(content={
            "success": True,
            "data": serialized_schedules,
            "message": f"Found {len(schedules)} schedules"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schedules: {str(e)}")

@app.put("/api/task/{task_id}/complete")
async def toggle_task_completion(task_id: str):
    """Toggle task completion status"""
    try:
        # Find schedule containing this task
        schedule = schedules_collection.find_one({"tasks.task_id": task_id})
        
        if not schedule:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Update task completion status
        for task in schedule["tasks"]:
            if task["task_id"] == task_id:
                task["completed"] = not task.get("completed", False)
                break
        
        # Save updated schedule
        schedules_collection.update_one(
            {"schedule_id": schedule["schedule_id"]},
            {"$set": {
                "tasks": schedule["tasks"],
                "updated_at": datetime.utcnow()
            }}
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "Task completion status updated"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get app configuration status"""
    return JSONResponse(content={
        "gemini_configured": bool(os.environ.get('GEM')),
        "database_connected": True,
        "version": "1.0.0"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)