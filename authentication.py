from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN
import os
from dotenv import load_dotenv
load_dotenv()
SECRET_TOKEN = os.getenv("SECRET_TOKEN")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != SECRET_TOKEN:
        print(credentials.credentials)
        print(SECRET_TOKEN)
        print(credentials.credentials)
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid Token"
        )
