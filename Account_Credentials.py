import os
import logging
from dotenv import load_dotenv
from schwabdev import Client  # Adjust the import based on your project structure

def main():
    """
    Main function to load Schwab credentials from .env,
    create a Client instance, and retrieve account hashes and numbers.
    """
    # 1. Load environment variables from the .env file
    load_dotenv()

    # 2. Fetch credentials from environment variables
    app_key = os.getenv("app_key")
    app_secret = os.getenv("app_secret")
    callback_url = os.getenv("callback_url", "https://127.0.0.1")
    tokens_file = os.getenv("token_path", "tokens.json")

    # Configure logging if desired (optional)
    logging.basicConfig(level=logging.INFO)

    # 3. Create the Client instance
    client = Client(
        app_key=app_key,
        app_secret=app_secret,
        callback_url=callback_url,
        tokens_file=tokens_file,
        timeout=5,
        update_tokens_auto=True
    )

    # 4. Call the account_linked() endpoint to retrieve account data
    response = client.account_linked()

    # Check if the request was successful
    if response.status_code == 200:
        # 5. Parse the JSON response
        try:
            data = response.json()
            print("Raw API Response:", data)  # Debugging: print the raw response

            # Debug the structure of the response
            if isinstance(data, list):
                print("Response is a list.")
                for item in data:
                    print(item)  # Inspect each item in the list
            elif isinstance(data, dict):
                print("Response is a dictionary.")
                linked_accounts = data.get("linkedAccounts", [])
                print("Linked Accounts:", linked_accounts)
            else:
                print("Unexpected response type:", type(data))
                return

            # Assuming the response is a list (based on your error)
            print("Linked Accounts:")
            for account in data:  # Loop through the list directly
                account_hash = account.get("accountHash")  # Adjust key names if necessary
                account_number = account.get("accountNumber")
                print(f"  - Account Number: {account_number}, Account Hash: {account_hash}")

        except Exception as e:
            print("Error while parsing the response:", e)

    else:
        # Log or raise an error if the response indicates a failure
        print(f"Failed to retrieve linked accounts. Status code: {response.status_code}")
        try:
            error_info = response.json()
            print("Error details:", error_info)
        except Exception as e:
            print("Could not parse error response:", e)


if __name__ == "__main__":
    main()
