
import os
import glob



def get_files(directory): 
    file_list = []
    for file in glob.glob(directory):
        file_list.append(file[15:])
        print(file[15:])
    file_list.sort()
    return file_list
        

def main():	
	file_list=get_files('./data/ext_sig/*')
	# event_list=['endpoint-malware','malicious-email','malicious-destination']
	event_list=['internal-armstrong']

	for i in event_list:
		for file in file_list:
			os.system('python3 eval_warning_replication_test.py -ext {} -method arimax -target armstrong -event {}'.format(file,i))

if __name__ == "__main__":
	import sys
	main()