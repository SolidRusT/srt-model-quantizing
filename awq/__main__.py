import sys
from app.main import main

if __name__ == "__main__":
    # Remove the script name from the arguments
    args = sys.argv[1:]
    
    if len(args) < 2:
        print("Usage: python -m awq <author> <model> [--quanter <quanter>]")
        sys.exit(1)
    
    author = args[0]
    model = args[1]
    quanter = None
    
    if len(args) > 3 and args[2] == "--quanter":
        quanter = args[3]
    
    main(author, model, quanter)