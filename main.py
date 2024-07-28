from data_preprocessing import load_data, preprocess_data
from model import IntentDataset, create_model
from train import train_model
from inference import get_response
import yaml

def main():
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load and preprocess data
    df, df_response = load_data()
    X_train, X_test, y_train, y_test, intent_to_index = preprocess_data(df)

    # Create datasets and model
    train_dataset, test_dataset = IntentDataset(X_train, y_train, config), IntentDataset(X_test, y_test, config)
    model = create_model(len(intent_to_index))

    # Train model
    train_model(model, train_dataset, test_dataset, config)

    # Interactive loop
    print("Enter 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response, intent = get_response(user_input, model, intent_to_index, df_response)
        print(f"BERT Bot: {response} (Intent: {intent})")
        print()

if __name__ == "__main__":
    main()