from fnmatch import fnmatch
#don't need re module since watchdog is separate
#from re import T
from flask import Flask, render_template, flash, redirect, request, session, logging, url_for
from forms import LoginForm, RegisterForm
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

import os
#from vidProcDbModels import session, VidLogs, User
from vidProcDbModels import dbSession, VidLogs, User

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = '!9m@S-dThyIlW[pHQbN^'

@app.route('/')
def home():
    return render_template('index.html')

'''
#FOR login, reg testing only REMOVE once checked.
import random
import string
 
def random_string_generator(str_size, allowed_chars):
    return ''.join(random.choice(allowed_chars) for x in range(str_size))

#test db queries for register nd login
@app.route('/test-reg/', methods = ['GET', 'POST'])
def test_login():
    #generate random login pwd etc.
    #pwd doesn't have to be unique
    chars = string.ascii_letters + string.punctuation
    num_of_chars = 8
    random_username = random_string_generator(num_of_chars, chars)
    random_email = random_username + '@dsds.com'
    mypwd = 'somepwd'
    hashed_password = generate_password_hash(mypwd, method='sha256')
    print ("pwd =", mypwd, "hashed pwd = ", hashed_password)
    new_user = User(
            name = random_username, 
            username = random_username, 
            email = random_email, 
            password = hashed_password )
    # saving user object into data base with hashed password
    dbSession.add(new_user)
    dbSession.commit()
    flash('You have successfully registered', 'success')
    # checking that user is exist or not by email
    #original
    #user = User.query.filter_by(email = form.email.data).first()
    user2 = dbSession.query(User).filter_by(email = new_user.email).first()
    print("user name = ", user2.name, "user email = ", user2.email)
    flash('You have successfully registered', 'success')
    # if registration successful, then redirecting to login Api
    return redirect(url_for('home'))
'''

# User Registration Api End Point
@app.route('/register/', methods = ['GET', 'POST'])
def register():
    # Creating RegistrationForm class object
    form = RegisterForm(request.form)
    # Cheking that method is post and form is valid or not.
    if request.method == 'POST' and form.validate():
        # if all is fine, generate hashed password
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        # create new user model object
        new_user = User(
            name = form.name.data, 
            username = form.username.data, 
            email = form.email.data, 
            password = hashed_password )
        # saving user object into data base with hashed password
        dbSession.add(new_user)
        dbSession.commit()
        flash('You have successfully registered', 'success')
        # if registration successful, then redirecting to login Api
        return redirect(url_for('login'))
    else:
        # if method is Get, than render registration form
        return render_template('register.html', form = form)
 
# Login API endpoint implementation
@app.route('/login/', methods = ['GET', 'POST'])
def login():
    #__bind_key__ = 'udb'
    # Creating Login form object
    form = LoginForm(request.form)
    # verifying that method is post and form is valid
    if request.method == 'POST' and form.validate:
        # checking that user is exist or not by email
        #original
        #user = User.query.filter_by(email = form.email.data).first()
        user = dbSession.query(User).filter_by(username = form.username.data).first()
        if user:
            # if user exist in database than we will compare our database hased password and password come from login form 
            if check_password_hash(user.password, form.password.data):
                # if password is matched, allow user to access and save email and username inside the session
                flash('You have successfully logged in.', "success")
                session['logged_in'] = True
                session['email'] = user.email 
                session['username'] = user.username
                session['name'] = user.name
                # After successful login, redirecting to home page
                return redirect(url_for('home'))
            else:
                # if password is in correct , redirect to login page
                flash('Username or Password Incorrect, Try Again', "Danger")
                return redirect(url_for('login'))
    # rendering login page
    return render_template('login.html', form = form)
 
@app.route('/logout/')
def logout():
    # Removing data from session by setting logged_flag to False.
    session['logged_in'] = False
    # redirecting to home page
    return redirect(url_for('home'))


@app.route('/main/<name>')
def hello_world(name):
    all_logs = dbSession.query(VidLogs).all()
    return render_template('main.html', all_logs = all_logs, name = name)
 
@app.route('/video/<sno>')
def video(sno):
    """        
    #you can show all available videos here one at a time if you want.
    log = dbSession.query(VidLogs).filter_by(sno = sno).all()
    for row in log:
        print ("sno: ",row.sno, "fname:",row.fname, "label:",row.label, "date_created",row.date_created)
        current_filename = row.fname
    """

    #but we only need one (assuming it is unique)
    #log = dbSession.query(VidLogs).filter_by(sno = sno).one()
    log = dbSession.query(VidLogs).filter_by(sno = sno).first()

    #if no records are found then set filename to empty - let teh template handle empty filename
    if log == None:
        current_filename = ''
        video_title = 'Record not found'
        return render_template('video.html', sno = int(sno), fname = current_filename, video_title=video_title)

    #record found
    current_filename = 'vids/' + log.fname
    video_title = log.label
    #get full file name to verify that the file actually exists in the directory!!!
    #Currently renaming an existing file or deleting existing files when watchdog is not running
    #will cause a mismatch..
    static_full_dirname = os.path.abspath('app/static')
    videos_full_dirname = static_full_dirname +  '/vids'
    full_file_name = videos_full_dirname + '/' + log.fname
    
    #get file name extension to check for file name extension - allow only mp4 files to be shown
    #just in case some .txt or some other files were also in that folder...
    #ideally, should use pathlib but os.path is already imported and good enough.
    #split gives back an array - 0th element is stuff before ext
    fname_parts = os.path.splitext(log.fname)
    file_ext = fname_parts[1]
    video_file_extensions = ['.webm', '.mp4', '.avi']
    #if the file is not there or the file ext is not .mp4, then
    #set fname to be empty and handle the empty filenames in the template itself
    #if (not os.path.isfile(full_file_name)) or (file_ext != '.webm' and file_ext != '.mp4'):
    if (not os.path.isfile(full_file_name)) or (file_ext not in video_file_extensions):
        print ("File not exist or not valid video file extension - ", full_file_name)
        current_filename = ''
    
    return render_template('video.html', sno = int(sno), fname = current_filename, video_title=video_title)
    

#Check the next commented-out route - it deletes log record and also the file from
#the directory but may be not a great idea to allow deleting files from the web users???
#unless the user is a very responsible admin...
@app.route('/delete/<name>/<sno>')
def delete(name,sno):
    dbSession.query(VidLogs).filter_by(sno = sno).delete()
    dbSession.commit()
    #show some feedback to user:
    flash('Log record successfully deleted..')
    #originally returning to root
    #return redirect('/')
    return redirect(url_for('hello_world',name = name))


"""
#FOLLOWING CODE WORKS. It will delete rec from the log and also from the dir.
#But... maybe not a good idea for anyone to delete files from your filesystem.. 
#So commenting out.

@app.route('/delete-from-logs-and-files/<sno>')
def delete_from_logs_and_files(sno):
    #but we only need one (assuming it is unique)
    #log = dbSession.query(VidLogs).filter_by(sno = sno).one()
    log = dbSession.query(VidLogs).filter_by(sno = sno).first()

    #if no records are found then set filename to empty - let teh template handle empty filename
    if log == None:
        current_filename = ''
        video_title = 'Record not found.. nothing to delete.'
        flash('Log record successfully deleted..')
        #originally returning to root
        #return redirect('/')        
        return redirect('/home')

    #record found
    current_filename = 'vids/' + log.fname
    video_title = log.label


    #now try from the filesystem
    #get full file name to verify that the file actually exists in the directory!!!
    #Currently renaming an existing file or deleting existing files when watchdog is not running
    #will cause a mismatch..
    static_full_dirname = os.path.abspath('static')
    videos_full_dirname = static_full_dirname +  '/vids'
    full_file_name = videos_full_dirname + '/' + log.fname
    
    #get file name extension to check for file name extension - allow only mp4 files to be shown
    #just in case some .txt or some other files were also in that folder...
    #ideally, should use pathlib but os.path is already imported and good enough.
    #split gives back an array - 0th element is stuff before ext
    fname_parts = os.path.splitext(log.fname)
    file_ext = fname_parts[1]
    
    #delete from db
    dbSession.query(VidLogs).filter_by(sno = sno).delete()
    dbSession.commit()
    flash('Log record deleted ..')
    #if the file is not there or the file ext is not .mp4, then
    #set fname to be empty and handle the empty filenames in the template itself
    if (os.path.isfile(full_file_name)) and (file_ext == '.mp4'):
        print ("File found - ", full_file_name, 'attemoting delete..')
        os.remove(full_file_name)
        print ("File deleted from filesystem - ", full_file_name)
        flash('File deleted ..')
        
    return redirect('/home')
"""

if __name__ == "__main__":
    app.run(debug=True)