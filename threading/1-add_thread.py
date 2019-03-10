import threading
import time

def T1_job():
    print('This is T1, id is :{}'.format(threading.current_thread()))
    for i in range(10):
        time.sleep(0.1)    
    print('T1,finished!')
    
def T2_job():
    print('This is T1, id is :{}'.format(threading.current_thread()))
    print('T2,finished!')
    
def main1():
    print(threading.active_count())
    print(threading.enumerate(),'\n')
    print(threading.current_thread())
    print('ok\n\n\n')
    T1 = threading.Thread(target=T1_job,name='T1')
    T2 = threading.Thread(target=T2_job,name='T2')
    T1.start()
    T2.start()
    
def main2():
    print(threading.active_count())
    print(threading.enumerate(),'\n')
    print(threading.current_thread())
    print('ok\n\n\n')
    T1 = threading.Thread(target=T1_job,name='T1')
    T2 = threading.Thread(target=T2_job,name='T2')
    T1.start()
    T2.start()
    T1.join()
    
    
if __name__=='__main__':
    main1()	
    main2()