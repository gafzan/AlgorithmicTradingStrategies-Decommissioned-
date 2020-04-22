from twilio.rest import Client
import os

account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
twilio_number = os.environ.get('MY_TWILIO_PHONE_NUMBER')
my_number = os.environ.get('MY_PHONE_NUMBER')


def send_sms(message: str)->None:
    # with the credential, create an instance of a twilio.rest Client
    client = Client(account_sid, auth_token)

    client.messages.create(
        to=my_number,
        from_=twilio_number,
        body=message
    )


if __name__ == '__main__':
    txt = input("Enter a text to send by SMS: ")
    send_sms(txt)
