# Save this as debug_hook.py in the same directory as your main.spec file

def debug_hook():
    import sys
    import os

    def excepthook(type, value, traceback):
        import traceback as tb
        print("An error occurred:")
        print("Type:", type)
        print("Value:", value)
        tb.print_tb(traceback)
        
        # Log to a file
        with open(os.path.join(os.path.dirname(sys.executable), 'error_log.txt'), 'a') as f:
            f.write("An error occurred:\n")
            f.write(f"Type: {type}\n")
            f.write(f"Value: {value}\n")
            tb.print_tb(traceback, file=f)
            f.write("\n\n")

    sys.excepthook = excepthook

debug_hook()
