import os 
import sys
import getpass

rootdir = "/" # Include path to repo

applications = {"VA"       : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/host_code -w 0 -e 1 -i #elements -x 0"],
                "GEMV"     : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/gemv_host -m #elements -n 2048"],
                "SpMV"     : ["NR_DPUS=X NR_TASKLETS=Y make all", "./bin/host_code -v 0 -f file_name"],
                "SEL"      : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/host_code -w 0 -e 1 -i #elements -x 0"],
                "UNI"      : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/host_code -w 0 -e 1 -i #elements -x 0"],
                "BS"       : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/bs_host -i #elements"],
                "TS"       : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/ts_host -n #elements"],
                "BFS"      : ["NR_DPUS=X NR_TASKLETS=Y make all", "./bin/host_code -v 0 -f file_name"],
                "MLP"      : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/mlp_host -m #elements -n 1024"],
                "NW"       : ["NR_DPUS=X NR_TASKLETS=Y BL=512 BL_IN=8 make all", "./bin/nw_host -w 0 -e 1 -n #elements"],
                "HST-S"    : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/host_code -w 0 -e 1 -b 256 -x 0"],
                "HST-L"    : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/host_code -w 0 -e 1 -b 256 -x 0"],
                "RED"      : ["NR_DPUS=X NR_TASKLETS=Y BL=Z VERSION=SINGLE make all", "./bin/host_code -w 0 -e 1 -i #elements -x 0"],
                "SCAN-SSA" : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/host_code -w 0 -e 1 -i #elements -x 0"],
                "SCAN-RSS" : ["NR_DPUS=X NR_TASKLETS=Y BL=Z make all", "./bin/host_code -w 0 -e 1 -i #elements -x 0"],
                "TRNS"     : ["NR_DPUS=X NR_TASKLETS=Y make all", "./bin/host_code -w 0 -e 1 -p #elements -o 12288 -x 0"],}

def run(app_name):
    
    NR_DPUS = [1, 4, 16, 64]
    NR_TASKLETS = [1, 2, 4, 8, 16]
    size = 1
    BL = [10] 
    if(app_name == "VA"):
        size = 2621440
    if(app_name == "GEMV"):
        size = 1024
    if(app_name == "SEL" or app_name == "UNI" or app_name == "SCAN-SSA" or app_name == "SCAN-RSS"):
        size = 3932160
    if(app_name == "TS"):
        size = 524288 
    if(app_name == "BS"):
        size = 262144
    if(app_name == "MLP"):
        size = 1024
    if(app_name == "RED"):
        size = 6553600
    if(app_name == "TRNS"):
        size = 1


    if app_name in applications:
        print ("------------------------ Running: "+app_name+"----------------------")
        print ("--------------------------------------------------------------------")
        if(len(applications[app_name]) > 1):
            make = applications[app_name][0]
            run_cmd = applications[app_name][1]
        
            os.chdir(rootdir + "/"+app_name)
            os.getcwd()
        
            os.system("make clean")

            try:
                os.mkdir(rootdir + "/"+ app_name +"/bin")
            except OSError:
                print ("Creation of the direction /bin failed")
                
            try:
                os.mkdir(rootdir + "/"+ app_name +"/log")
            except OSError:
                print ("Creation of the direction /log failed")
            
            try:
                os.mkdir(rootdir + "/"+ app_name +"/log/host")
            except OSError: 
                print ("Creation of the direction /log/host failed")

            try:
                os.mkdir(rootdir + "/"+ app_name +"/profile")
            except OSError:
                print ("Creation of the direction /profile failed")
        

            for r in NR_DPUS:
                for t in NR_TASKLETS:
                    for b in BL:
                        m = make.replace("X", str(r))
                        m = m.replace("Y", str(t))
                        m = m.replace("Z", str(b))
                        print ("Running = " + m) 
                        try:
                            os.system(m)
                        except: 
                            pass 

                        if(app_name == "NW"):
                            if(r == 1):
                                r_cmd = run_cmd.replace("#elements", str(512))
                            if(r == 4):
                                r_cmd = run_cmd.replace("#elements", str(2048))
                            if(r == 16):
                                r_cmd = run_cmd.replace("#elements", str(8192))
                            if(r == 64):
                                r_cmd = run_cmd.replace("#elements", str(32768))
                        elif(app_name == "GEMV" or app_name == "MLP" or app_name == "TS" or app_name == "BS"):
                            r_cmd = run_cmd.replace("#elements", str(r * size))
                        else:
                            r_cmd = run_cmd.replace("#elements", str(size))
                        if(app_name == "BFS"):
                            if(r == 1):
                                # Generate rMat graphs using:
                                # https://github.com/cmuparlay/pbbsbench/blob/master/testData/graphData/rMatGraph.html
                                # https://github.com/cmuparlay/pbbsbench/blob/master/testData/graphData/rMatGraph.C
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - rMat graph
                            if(r == 4):
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - rMat graph
                            if(r == 16):
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - rMat graph
                            if(r == 64):
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - rMat graph
                        if(app_name == "SpMV"):
                            if(r == 1):
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - Check SpMV/data/generate
                            if(r == 4):
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - Check SpMV/data/generate
                            if(r == 16):
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - Check SpMV/data/generate
                            if(r == 64):
                                r_cmd = run_cmd.replace("file_name", "/") # Include path to input file - Check SpMV/data/generate
                        r_cmd = r_cmd +  " >> profile/out_tl"+str(t)+"_bl"+str(b)+"_dpus"+str(r) 
                        
                        print ("Running = " + app_name + " -> "+ r_cmd)
                        try:
                            os.system(r_cmd) 
                        except:  
                            pass 
        else:
            make = applications[app_name] 

            os.chdir(rootdir + "/"+app_name)
            os.getcwd()
        
            try:
                os.mkdir(rootdir + "/"+ app_name +"/bin")
                os.mkdir(rootdir + "/"+ app_name +"/log")
                os.mkdir(rootdir + "/"+ app_name +"/log/host")
                os.mkdir(rootdir + "/"+ app_name +"/profile")
            except OSError:
                print ("Creation of the direction failed")

            print (make)    
            os.system(make + ">& profile/out")

    else:
        print ( "Application "+app_name+" not available" )

def main():
    if(len(sys.argv) < 2):
        print ("Usage: python run.py application")
        print ("Applications available: ")
        for key, value in applications.items():
            print (key )
        print ("All")

    else:
        cmd = sys.argv[1]
        print ("Application to run is: " + cmd )
        if cmd == "All":
            for key, value in applications.items():
                run(key)
                os.chdir(rootdir)
        else:
            run(cmd)

if __name__ == "__main__":
    main()
