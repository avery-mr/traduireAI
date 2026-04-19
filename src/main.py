from datasets import load_dataset
from baseline_model import translate_baseline
from transformer_model import translate_transformer
from evaluate import evaluate_models

def main():
    while True:
        print("\nTraduireAI")
        print("1. Translate Text")
        print("2. Run BLEU Evaluation")
        print("3. Quit")

        choice = input("\nEnter your choice: ")
        if choice == "1":
            text = input("Enter English text: ")
            print("\nBaseline: ", translate_baseline(text))
            print("\nMarianMT: ", translate_transformer(text))

        elif choice == "2":
            print("\nEvaluate on...")
            print("(a) full test set (takes a long time....)")
            print("(b) defined subset size (less than 2000)")
            mode = input("\nEnter your choice: ").strip().lower()

            if mode == "a":
                evaluate_models()
            if mode == "b":
                n = int(input("\nEnter subset size (0 to 2000): "))
                if n > 2000:
                    print("\nSubset size cannot be greater than or equal to 2000.")
                else:
                    evaluate_models(sample_size=n)
            else:
                print("Invalid Selection...")

        elif choice == "3":
            print("\nA la prossima!")
            break

        else:
            print("Invalid Selection...")



if __name__ == "__main__":
    main()