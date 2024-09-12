from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import watchdog.events
import watchdog.observers
import time

#word session may conflict with db session?
#from vidProcDbModels import session, VidLogs, User
from vidProcDbModels import dbSession, VidLogs, User


filepath = ""

def on_created(event):
    print(f"{event.src_path} has been created.")
    fname = event.src_path.rsplit('\\',1)[1]
    label = fname.rsplit('.')[0]
    vidlog = VidLogs(fname = fname, label = label)
    dbSession.add(vidlog)
    dbSession.commit()  
 
def on_deleted(event):
    print(f"Someone deleted {event.src_path}!")
    fname = event.src_path.rsplit('\\',1)[1]
    #Original using flask_alchemy was - VidLogs.query.filter_by(fname = fname).delete()
    #make sure that the fname is unique!!!
    dbSession.query(VidLogs).filter_by(fname = fname).delete()
    dbSession.commit()



if __name__ == "__main__":
    patterns = ["*"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
    my_event_handler.on_created = on_created
    my_event_handler.on_deleted = on_deleted
    path = 'app/static'
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)
    my_observer.start()
    print("Watchdog STARTED. Will keep checking for changes...")
    try:
        while True:                
            time.sleep(1)
    except:
        my_observer.stop()
        print("Watchdog STOPPED")

#keep this thread "blocked" to avoid hassles with multiple threads running
    my_observer.join()


 #pathlib