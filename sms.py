from twilio.rest import Client
from secrets import account_sid, auth_token, twilio_number, my_number


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
