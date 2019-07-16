#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Mohammed Assiri
# DATE CREATED: July 7, 2017
# REVISED DATE:    ==         <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print 
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()
    
    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    
    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"+str(int((tot_time%3600)%60)))



# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    parser.add_argument('--dir', type=str, default='pet_images/', help='path to folder of images')
    parser.add_argument('--arch', type=str, default='vgg', help='chosen model')
    parser.add_argument('--dogfile', type=str, default='dognames.txt', help='text file that has dognames')
    return parser.parse_args()



def get_pet_labels(image_dir):
    in_files = listdir(image_dir)
    petlabels_dic = dict()
    for idx in range(0, len(in_files), 1):
        if in_files[idx][0] != ".":
            image_name = in_files[idx].split("_")
            pet_label = ""
            for word in image_name:
                if word.isalpha():
                    pet_label += word.lower() + " "
                    pet_label = pet_label.strip()
                    
            if in_files[idx] not in petlabels_dic:
                petlabels_dic[in_files[idx]] = pet_label
            else:
                print("Warning: Duplicate files exist in directory", in_files[idx])
                        
    return(petlabels_dic)
                        



def classify_images(images_dir, petlabel_dic, model):
    results_dic = dict()
    for key in petlabel_dic:
        model_label = classifier(images_dir+key, model)
        model_label = model_label.lower()
        model_label = model_label.strip()
        truth = petlabel_dic[key]
        found = model_label.find(truth)
        if found >= 0:
            if ( (found == 0 and len(truth)==len(model_label)) or (((found == 0) or (model_label[found - 1] == " ")) and ((found + len(truth) == len(model_label)) or (model_label[found + len(truth): found+len(truth)+1] in (","," "))))):
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 1]
                else:
                    if key not in results_dic:
                        results_dic[key] = [truth, model_label, 0]
                    else:
                        if key not in results_dic:
                            results_dic[key] = [truth, model_label, 0]
               
    return(results_dic)




def adjust_results4_isadog(results_dic, dogsfile):
    dognames_dic = dict()
    with open(dogsfile, "r") as infile:
        line = infile.readline()
        while line != "":
            line = line.rstrip()
            if line not in dognames_dic:
                dognames_dic[line] = 1
            else:
                print("**Warning: Duplicate dognames", line)            
                line = infile.readline()
    
    for key in results_dic:
        if results_dic[key][0] in dognames_dic:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1, 1))
            else:
                results_dic[key].extend((1, 0))
        else:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((0, 1))
            else:
                results_dic[key].extend((0, 0))




def calculates_results_stats(results_dic):
    results_stats=dict()
    results_stats['n_dogs_img'] = 0
    results_stats['n_match'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_notdogs'] = 0
    results_stats['n_correct_breed'] = 0       
    for key in results_dic:
        if results_dic[key][2] == 1:
            results_stats['n_match'] += 1
        if sum(results_dic[key][2:]) == 3:
                results_stats['n_correct_breed'] += 1
        if results_dic[key][3] == 1:
            results_stats['n_dogs_img'] += 1
            if results_dic[key][4] == 1:
                results_stats['n_correct_dogs'] += 1
        else:
            if results_dic[key][4] == 0:
                results_stats['n_correct_notdogs'] += 1
                
    results_stats['n_images'] = len(results_dic)
    results_stats['n_notdogs_img'] = (results_stats['n_images'] - results_stats['n_dogs_img']) 
    results_stats['pct_match'] = (results_stats['n_match'] / results_stats['n_images'])*100.0
    results_stats['pct_correct_dogs'] = (results_stats['n_correct_dogs'] / results_stats['n_dogs_img'])*100.0    
    results_stats['pct_correct_breed'] = (results_stats['n_correct_breed'] / results_stats['n_dogs_img'])*100.0
    
    if results_stats['n_notdogs_img'] > 0:
        results_stats['pct_correct_notdogs'] = (results_stats['n_correct_notdogs'] / results_stats['n_notdogs_img'])*100.0

    else:
        results_stats['pct_correct_notdogs'] = 0.0
                                                
    return results_stats



def print_results(results_dic, results_stats, model, print_incorrect_dogs = False, print_incorrect_breed = False):
    print("\n\n*** Results Summary for CNN Model Architecture",model.upper(), "***")
    print("%20s: %3d" % ('N Images', results_stats['n_images']))
    print("%20s: %3d" % ('N Dog Images', results_stats['n_dogs_img']))
    print("%20s: %3d" % ('N Not-Dog Images', results_stats['n_notdogs_img']))
    print(" ")
    for key in results_stats:
        if key[0] == "p":
            print("%20s: %5.1f" % (key, results_stats[key]))
                                                
    if (print_incorrect_dogs and ((results_stats['n_correct_dogs'] + results_stats['n_correct_notdogs'])!= results_stats['n_images'])):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for key in results_dic:
            if sum(results_dic[key][3:]) == 1:
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0], results_dic[key][1]))

    if (print_incorrect_breed and (results_stats['n_correct_dogs'] != results_stats['n_correct_breed'])):
        print("\nINCORRECT Dog Breed Assignment:")
        for key in results_dic:
            if ( sum(results_dic[key][3:]) == 2 and
                results_dic[key][2] == 0 ):
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0],
                                                          results_dic[key][1]))
                


                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
