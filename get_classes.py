import subprocess

def main():
    try:
        out = subprocess.check_output(["git", "show", "8959eca:genetic_algorithm.py"], encoding="utf-8")
        
        start_ii = out.find("class MSRSIITargetedMutation")
        start_iv = out.find("class MSRSIVTargetedMutation")
        end_iv = out.find("def run_nsga2_optimization_msrs_ii")
        
        print("start_ii:", start_ii)
        print("start_iv:", start_iv)
        print("end_iv:", end_iv)
        
        if start_ii != -1 and end_iv != -1:
            extracted = out[start_ii:end_iv]
            with open("extracted_classes.txt", "w", encoding="utf-8") as f:
                f.write(extracted)
            print("Successfully wrote extracted_classes.txt")
        else:
            print("Could not find the class definitions in the file.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
