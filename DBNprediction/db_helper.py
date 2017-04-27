'''
Created on Nov 8, 2016

@author: atri
'''


import pymysql
import configparser
import sqlite3

def connect():
    '''config = configparser.ConfigParser()
    cf = config.read('config/connection.ini')
    cf_default = config['DEFAULT']
    db = pymysql.connect( cf_default['host'],cf_default['user'],cf_default['pass'],cf_default['db'])'''
    db = sqlite3.connect('/home/atri/sharcnet/trajectories.db')
    return db

def execute(db,string):
    cursor = db.cursor()
    cursor.execute(string)
    data = cursor.fetchone()

