#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/27 15:25
# @Author  : Aries
# @Site    : 
# @File    : test.py
# @Software: PyCharm Community Edition

def power(x,n):
    s=1
    while n>0:
        s=s*x
        n=n-1
    return s

def power(x,n=2):
    s=1
    while n>0:
        s=s*x
        n=n-1
    return s

def add_end(L=None):
    if L is None:
        L=[]
    L.append('End')
    return L

def multi(*number):
    sum=0
    for n in number:
        sum=sum+n
    return sum

def product(*z):
    sum=1
    for i in z:
        sum=sum*i
    return sum
