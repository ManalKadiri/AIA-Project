import requests
import sys
import os

def send_email(subject, body):
    api_key = os.getenv("POSTMARK_API_KEY")  # Récupère la clé depuis les variables d'environnement
    if not api_key:
        print("POSTMARK_API_KEY not found!")
        return

    url = "https://api.postmarkapp.com/email"
    headers = {
        "X-Postmark-Server-Token": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "From": "o.elkenfaoui@cabinet-conex.fr",  # Remplacez par l'adresse autorisée sur Postmark
        "To": "contact@cabinet-conex.fr",  # Destinataire de l'email
        "Subject": subject,
        "TextBody": body
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print("Email sent successfully.")
        else:
            print(f"Failed to send email. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Error while sending email: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python send_email.py <subject> <body>")
        sys.exit(1)

    subject = sys.argv[1]
    body = sys.argv[2]
    send_email(subject, body)
