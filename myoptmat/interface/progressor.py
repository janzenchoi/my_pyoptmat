"""
 Title:          Progress
 Description:    Contains progress visualisation functions
 Author:         Janzen Choi

"""

# Libraries
import time

# The Progress Visualiser class
class ProgressVisualiser:

    # Constructor
    def __init__(self, num_steps, pretext="", posttext="", clear=False, newline=True, quiet=False):
        
        # Initialise inputs
        self.num_steps  = num_steps
        self.pretext    = pretext
        self.posttext   = posttext
        self.clear      = clear
        self.newline    = newline
        self.quiet      = quiet
        
        # Initialise other
        self.start_time = time.time()
        self.display_string = ""
        
        # First print
        self.curr_step = 1
        self.__print_progress__()

    # Progresses the process
    def progress(self, pretext=None, posttext=None):
        self.pretext = pretext if pretext != None else self.pretext
        self.posttext = posttext if posttext != None else self.posttext
        if self.curr_step <= self.num_steps:
            self.__print_progress__() 
        if self.curr_step == self.num_steps and not self.clear and self.newline:
            print("")
        if self.curr_step == self.num_steps and self.clear:
            self.__clear_display__()
        self.curr_step += 1

    # Prematurely ends the process
    def end(self):
        while self.curr_step <= self.num_steps:
            self.progress()

    # Clears the display string
    def __clear_display__(self):
        if not self.quiet:
            print("\b" * (len(self.display_string)), end="", flush=True)

    # Prints the progress
    def __print_progress__(self):
        
        # For no printing
        if self.quiet:
            return
        
        # Clear previous visual and apply pretext
        self.__clear_display__()
        self.display_string = f"{self.pretext}" if self.pretext != "" else ""
        
        # Write progress
        self.display_string += f"iters={self.curr_step}/{self.num_steps}, "
        self.display_string += f"time={round(time.time() - self.start_time, 1)}s"
        
        # Apply post string and print
        self.display_string += self.posttext
        print(self.display_string, end="", flush=True)
