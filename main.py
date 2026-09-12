from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
import os

app = FastAPI()

# 1. Serve your main login/home page at the root ("/")
@app.get("/")
def read_root():
    # Replace 'my_gpt.html' with the exact name of your login HTML file
    if os.path.exists("my_gpt.html"):
        return FileResponse("my_gpt.html")
    return {"error": "Home page HTML file not found"}

# 2. Protected Mining Portal Route (Loads your dashboard HTML if logged in)
@app.get("/mining-portal")
def get_mining_portal(request: Request):
    # Check if the user session cookie exists
    user_token = request.cookies.get("miner_session")
    
    if not user_token:
        # Redirect unauthenticated users back to the home/login page
        return RedirectResponse(url="/", status_code=303)
        
    # Replace 'mining-portal.html' with the actual filename of your dashboard
    if os.path.exists("mining-portal.html"):
        return FileResponse("mining-portal.html")
    
    # Fallback message if the HTML file isn't created yet
    return {"message": "Welcome to the authorized manganese mining portal!"}

# 3. Login Endpoint (Sets the cookie upon successful login)
class LoginCredentials(BaseModel):
    email: str
    password: str

@app.post("/api/login")
def login_contractor(credentials: LoginCredentials, response: Response):
    if credentials.email == "miner@manganese.com" and credentials.password == "securepassword123":
        response.set_cookie(key="miner_session", value="authorized_token", httponly=True)
        return {"status": "success"}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")
