from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(username=account_sid, password=auth_token)

def send_alert(msg: str):
    client.messages.create(
        from_='whatsapp:+14155238886',
        body=msg,
        to='whatsapp:+17328047031'
    )

if __name__ == "__main__":
    send_alert("Hello, this is a test message from Twilio!")