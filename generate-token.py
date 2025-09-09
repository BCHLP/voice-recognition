import jwt
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

if __name__ == "__main__":

    load_dotenv()

    payload = {
            'exp': datetime.now() + timedelta(days=3650)
    }
    print(jwt.encode(payload, os.getenv('JWT_TOKEN'), algorithm='HS256'))