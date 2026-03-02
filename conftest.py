"""root conftests adds IT_Newsreader/ to sys.path so all absolute imports work"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
