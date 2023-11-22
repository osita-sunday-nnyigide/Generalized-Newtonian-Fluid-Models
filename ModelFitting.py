#!/usr/bin/env python

__doc__ = """

This program requires python 3.6 or higher.

This module has the function(s) that is used to 

fit data to the GNF Models.

"""

__author__     = "Osita Sunday Nnyigide"

__copyright__  = "Copyright 2022, Osita Sunday Nnyigide"

__credits__    = ["Hyun Kyu"]

__license__    = "MIT"

__version__    = "1.0.0"

__maintainer__ = "Osita Sunday Nnyigide"

__email__      = "osita@protein-science.com"

__status__     = "Production"

__date__       = "November 22, 2023"

import os
import customtkinter
import sys
import numpy as np
import tkinter as tk
from tkinter import *
from scipy import optimize
from tkinter import filedialog
from matplotlib import gridspec
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class GeneralizedNeutonianFluidModels:

    def PowellEyringModel(self, x, eta_0, eta_inf, lbda):
        return eta_inf + (eta_0 - eta_inf) * (np.arcsinh(lbda * x) / (lbda * x))

    def EllisModel(self, x, eta_0, eta_inf, lbda, a) : 
        return eta_inf + ((eta_0 - eta_inf)/ (1 + (lbda * x) ** a))

    def SiskoModel(self, x, eta_inf, lbda, n) :
        return eta_inf + (lbda*(x**n-1))

    def WilliamsonModel(self, x, eta_0, lbda, n) :
        return eta_0/ (1 + (lbda * x) ** n)

    def CrossModel(self, x , eta_0, eta_inf, lbda, a):
        return eta_inf + ((eta_0 - eta_inf)/ (1 + (lbda * x) ** a))

    def PowerLawModel(self, params, x_data, y_data):
        K, n = params
        y_predicted = K * x_data**(n-1)
        error = np.sum((y_data - y_predicted)**2)  # You can use different error metrics as needed
        return error

    def CarreauYasudaModel(self, params, x, y):
        eta_0, eta_inf, lbda, a, n =params
        y_predicted = eta_inf + (eta_0 - eta_inf) * (1 + (lbda * x) ** a) ** ((n - 1) / a)
        error=np.sum((y - y_predicted)**2)
        return error

    def BinghamModel(self, shear_rate, tau0, K):
        return tau0 + K * shear_rate

    def CassonModel(self, shear_rate, tau0, K):
        return np.sqrt(tau0) + np.sqrt(K * shear_rate)

    def HerschelBulkleyModel(self, shear_rate, tau0, K, n):
        return tau0 + K * shear_rate**n

    def FitPowellEyringModel(self, x, y, μo=3354.07, μf=42.2583, λ=2.68884e-5):
        bounds = ([min(y), 0, -np.inf], [max(y), min(y), np.inf])
        popt, pcov = optimize.curve_fit(self.PowellEyringModel, x, y, p0=[μo , μf, λ])
        fitted_y=self.PowellEyringModel(x, *popt)

        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity [Pa.s]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

        plt.text(min(x),min(y)*2,'Newtonian viscosity={:.3f}\nInfinite viscosity={:.3f}\nConsistency={:.3f}\nRsqr={:.3f}'.format\
                (popt[0], popt[1], popt[2], R_squared)) 

        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitSiskoModel(self, x, y, μf=42.2583, λ=2.68884e-5, n=1):
        popt, pcov = optimize.curve_fit(self.SiskoModel, x, y, p0=[μf, λ, n])
        fitted_y=self.SiskoModel(x, *popt)

        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity [Pa.s]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

        plt.text(min(x),min(y)*2,'Infinite viscosity={:.3f}\nConsistency={:.3f}\nPower law index={:.3f}\nRsqr={:.3f}'.format(popt[0], popt[1], popt[2], R_squared)) 
        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitWilliamsonModel(self, x, y, μf=3354.07, λ=2.68884e-5, n=-1945.61):
        popt, pcov = optimize.curve_fit(self.WilliamsonModel, x, y, p0=[μf, λ, n])
        fitted_y=self.WilliamsonModel(x, *popt)

        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity [Pa.s]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

        plt.text(min(x),min(y)*2,'Infinite viscosity={:.3f}\nConsistency={:.3f}\nPower law index={:.3f}\nRsqr={:.3f}'.format(popt[0], popt[1], popt[2], R_squared)) 
        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitEllisModel(self, x, y, μo=3354.07, μf=42.2583, λ=2.68884e-5, n=0.902192):
        popt, pcov = optimize.curve_fit(self.EllisModel, x, y, p0=[μo , μf, λ, n])
        fitted_y=self.EllisModel(x, *popt)

        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear stress [Pa]",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity [Pa.s]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

        plt.text(min(x),min(y)*2,'Newtonian viscosity={:.3f}\nInfinite viscosity={:.3f}\nConsistency={:.3f}\nPower law index={:.3f}\nRsqr={:.3f}'.format\
                (popt[0], popt[1], popt[2], popt[3], R_squared)) 

        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitCrossModel(self, x, y, μo=3354.07, μf=42.2583, λ=2.68884e-5, n=0.902192):
        bounds = ([min(y), 0, -np.inf, 0], [max(y), np.inf, np.inf, 1])
        popt, pcov = optimize.curve_fit(self.CrossModel, x, y, p0=[μo , μf, λ, n])
        fitted_y=self.CrossModel(x, *popt)

        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity [Pa.s]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

        plt.text(min(x),min(y)*2,'Zero shear viscosity={:.3f}\nInfinite viscosity={:.3f}\nConsistency={:.3f}\nPower law index={:.3f}\nRsqr={:.3f}'.format\
                (popt[0], popt[1], popt[2], popt[3], R_squared)) 

        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitCarreauYasudaModel(self, x, y, μo=3354.07, μf=42.2583, λ=2.68884e-5, a=0.902192, n=-1945.61):
        initial_guess = [μo , μf, λ, a, n]  # Initial guesses for K and n       
        bounds = [(-np.inf, np.inf), (y[-1], np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
        result = optimize.minimize(self.CarreauYasudaModel, initial_guess, args=(x, y), bounds=bounds)
        opt_params = result.x 
        eta_0, eta_inf, lbda, a, n = opt_params
        fitted_y = eta_inf + (eta_0 - eta_inf) * (1 + (lbda * x) ** a) ** ((n - 1) / a)
        y_mean = np.mean(y)

        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity [Pa.s]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)
        plt.text(min(x),min(y)*2,'Zero shear viscosity={:.3f}\nInfinite viscosity={:.3f}\nConsistency={:.3f}\nTransition parameter={:.3f}\nPower law index={:.3f}\nRsqr={:.3f}'.format\
                (eta_0, eta_inf, lbda, a, n, R_squared))

        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitPowerLawModel(self, x, y, k=1, n=1):
        initial_guess = [k, n]  # Initial guesses for K and n
        bounds = [(-np.inf, np.inf), (y[-1], np.inf)]
        result = optimize.minimize(self.PowerLawModel, initial_guess, args=(x, y), bounds=None)
        opt_params = result.x 
        K_opt, n_opt = opt_params
        fitted_y = K_opt * x**(n_opt - 1.0)

        y_mean = np.mean(y)
        SST = np.sum((y - y_mean)**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])

        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Viscosity [Pa.s]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)
        plt.text(min(x),min(y)*2,"Consistency={:.3f}\nPower law index={:.3f}\nRsqr={:.3f}\n".format(K_opt, n_opt, R_squared))
        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitBinghamModel(self, x, y, τ=2.0, μo=1.0):
        popt, pcov = optimize.curve_fit(self.BinghamModel, x, y, p0=[τ , μo])
        fitted_y=self.BinghamModel(x, *popt)
        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Shear stress [Pa]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)
        plt.text(max(x)/2,min(y)*2,'Yield stress={:.3f}\nPlastic viscosity={:.3f}\nRsqr={:.3f}'.format(popt[0], popt[1], R_squared))

        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitHerschelBulkleyModel(self, x, y, τo=1.0, k=1.0, n=0.5): 
        bounds = ([0.5, -np.inf, -np.inf], [np.inf, np.inf, np.inf])             
        popt, pcov = optimize.curve_fit(self.HerschelBulkleyModel, list(x), list(y), p0=[τo, k, n], bounds=bounds)
        fitted_y=self.HerschelBulkleyModel(x, *popt)
        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Shear stress [Pa]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)
        plt.text(max(x)/2,min(y)*2,'Yield stress={:.3f}\nConsistency={:.3f}\nFlow index={:.3f}\nRsqr={:.3f}'.format(popt[0], popt[1], popt[2], R_squared))

        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

    def FitCassonModel(self, x, y, τo=1, μo=1):
        bounds = ([-np.inf, 0.0], [(max(y)), np.inf])
        popt, pcov = optimize.curve_fit(self.CassonModel, x, y, p0=[τo, μo])
        fitted_y=self.CassonModel(x, *popt)
        SST = np.sum((y - np.mean(y))**2)
        SSE = np.sum((y - fitted_y)**2)
        R_squared = 1 - (SSE / SST)

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])
        plt.scatter(x, y, label='Data')
        ax1.plot(x, fitted_y, '--', color ='red', label ="Model fitting")
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_xlabel("Shear rate [1/s]",family="serif",  fontsize=12)
        ax1.set_ylabel("Shear stress [Pa]",family="serif",  fontsize=12)
        ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
        ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)
        plt.text(max(x)/2,min(y)*2,'Yield stress={:.3f}\nCasson viscosity={:.3f}\nRsqr={:.3f}'.format(popt[0], popt[1], R_squared))

        plt.legend()
        fig.tight_layout()
        fig.savefig("fitted_data.png", format="png",dpi=300, bbox_inches='tight')
        plt.show()

class GraphicalUserInterface(GeneralizedNeutonianFluidModels):

    def __init__(self):

        self.shear_rate=[] 
        self.viscosity=[]
        self.default=False
        self.GUI = tk.Tk()
        self.width= self.GUI.winfo_screenwidth()
        self.height= self.GUI.winfo_screenheight()
        self.GUI.geometry("%dx%d" % (self.width, self.height))
        self.GUI.title('Complex Fluid Laboratory') # give it a title
        self.GUI.wm_attributes('-toolwindow', 'False')
        self.GUI.state('zoomed')   
             
        self.frame = customtkinter.CTkFrame(self.GUI, width=self.width, fg_color='#50bfab')
        self.frame.pack(pady=0, padx=0, fill="both", expand=True, )

        for i in range(20):
            self.frame.columnconfigure(i, weight=1)
        self.frame.rowconfigure(0, weight=0)

        self.data_x_axis = tk.Text(self.frame, width=20, bg='#F5D0C7', height=20, wrap=WORD)
        self.data_y_axis = tk.Text(self.frame, width=20, bg='#F5D0C7', height=20, wrap=WORD)

        self.zero_vis = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.inf_vis  = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.lamda    = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.trans    = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.pw_indx  = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)

        self.zero_vis_label  = tk.Label(self.frame, text='μo=', width=2, font='none 11', bg='#50bfab')
        self.inf_vis_label   = tk.Label(self.frame, text='μf=', width=2, font='none 11', bg='#50bfab')
        self.lamda_label     = tk.Label(self.frame, text='λ=',  width=2, font='none 11', bg='#50bfab' )
        self.trans_label     = tk.Label(self.frame, text='a=',  width=2, font='none 11', bg='#50bfab')
        self.pw_indx_label   = tk.Label(self.frame, text='n=',  width=2, font='none 11', bg='#50bfab')

        self.upload_label    = tk.Label(self.frame, text='Upload data or paste X and Y',  font='none 12 bold', bg='#50bfab')
        self.upload_txtentry = tk.Entry(self.frame, width=50,)
        self.upload_txtentry.bind("<Button-1>", lambda e: self.upload_txtentry.delete(0, tk.END))
        self.upload_txtentry.insert(0, "                         *.dat, *.csv or *.txt only")

        self.upload_btn = tk.Button(self.frame, text='Browse', width=7, font='none 12 bold',command=self.osPath, fg='black')

        options = [ "PowellEyring",              
                    "HerschelBulkley",
                    "Carreau-Yasuda",
                    "Williamson",
                    "Power-Law",
                    "Bingham", 
                    "Casson",                                         
                    "Cross",
                    "Sisko",
                    "Ellis",
                  ]

        self.model = StringVar(self.frame)
        self.model.set("Select Model")

        self.drop_menu  = tk.OptionMenu(self.frame, self.model,*options, command=self.option_handle)

        self.submit_btn = tk.Button(self.frame, text='Submit', width=7, font='none 12 bold', command=self.PlotData)
        self.exit_btn   = tk.Button(self.frame, text='Exit', bg='red',width=7, font='none 12 bold', command=self.close_window)

        self.upload_btn.grid(row=2, column=9, padx=30, pady=5, ipady=7, sticky=W)
        self.upload_label.grid(row=1, column=9, padx=45, pady=5)
        self.upload_txtentry.grid(row=2, column=9, padx=90, pady=5, ipady=7)
        self.drop_menu.grid(row=2, column=9, padx=0, pady=0, sticky=E)

        self.data_x_axis.grid(row=3, column=9, padx=150, pady=0, ipady=7, sticky=W)
        self.data_y_axis.grid(row=3, column=9, padx=0, pady=0, ipady=7,sticky=E)

        self.zero_vis.grid(row=4, column=9, padx=120, pady=5, ipady=5, sticky=W)
        self.zero_vis_label.grid(row=4, column=9, padx=85, pady=5, ipady=0, sticky=W)

        self.inf_vis.grid(row=4, column=9, padx=240, pady=5, ipady=5,sticky=W)
        self.inf_vis_label.grid(row=4, column=9, padx=205, pady=5, ipady=0,sticky=W)

        self.lamda.grid(row=4, column=9, padx=200, pady=5, ipady=5,sticky=E)
        self.lamda_label.grid(row=4, column=9, padx=250, pady=5, ipady=0,sticky=E)

        self.trans.grid(row=4, column=9, padx=100, pady=5, ipady=5,sticky=E)
        self.trans_label.grid(row=4, column=9, padx=150, pady=5, ipady=0,sticky=E)

        self.pw_indx.grid(row=4, column=9, padx=0, pady=5, ipady=5,sticky=E)
        self.pw_indx_label.grid(row=4, column=9, padx=50, pady=5, ipady=0,sticky=E)

        self.cons_indx   = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.plaw_const  = tk.Text(self.frame, width=5, bg='#F5D0C7', height=1, wrap=WORD)
        self.cons_indx_label   = tk.Label(self.frame, text='K=', width=2, font='none 11', bg='#50bfab' )
        self.plaw_const_label  = tk.Label(self.frame, text='n=',  width=2, font='none 11',bg='#50bfab' )

        self.data_x_axis.bind("<Button-1>", lambda e: self.data_x_axis.delete(0.0, END))
        self.data_y_axis.bind("<Button-1>", lambda e: self.data_y_axis.delete(0.0, END))
        self.data_x_axis.insert(END, 'Enter X data here') # END for Text and 0 for Entry
        self.data_y_axis.insert(END, 'Enter Y data here')

        self.zero_vis.bind("<Button-1>", lambda e: self.zero_vis.delete(0.0, END))
        self.inf_vis.bind("<Button-1>", lambda e: self.inf_vis.delete(0.0, END))
        self.lamda.bind("<Button-1>", lambda e: self.lamda.delete(0.0, END))
        self.trans.bind("<Button-1>", lambda e: self.trans.delete(0.0, END))
        self.pw_indx.bind("<Button-1>", lambda e: self.pw_indx.delete(0.0, END))

        self.submit_btn.grid(row=5, column=9, padx=200, pady=5,ipady=7,sticky=W)
        self.exit_btn.grid(row=5, column=9, padx=40, pady=5,ipady=7,sticky=E)
        self.data_x_axis.focus()
        self.data_y_axis.focus()

        self.zero_vis.focus()
        self.inf_vis.focus()
        self.lamda.focus()
        self.trans.focus()
        self.pw_indx.focus()

        self.make_textmenu()
        self.GUI.bind_class("Text", "<Button-3><ButtonRelease-3>", self.show_textmenu)
        self.GUI.bind_class("Text", "<Control-a>", self.callback_select_all)
        self.GUI.mainloop()

    def option_handle(self, selected):
        try:
            self.pop.destroy()
        except:
            pass
        self.fitting_param(selected)

    def open_popup(self):
       self.pop= Toplevel(self.GUI)
       self.pop.geometry("300x100")
       self.pop.title("Report Window")
       Label(
                self.pop,text="Optimization failed." 
                " Check your data or\nChange fitting parameters.",
                font=('none 11 bold'),justify=LEFT).place(x=5,y=5
            )

    def fitting_param(self, model):

        CarreauYasudaModel="Enter values for [μo, μf, λ, a, n] as \nthe zero shear viscosity, infinite \nviscosity, consistency, transition \nand power law index, respectively"

        HerschelBulkleyModel="Enter values for [μo, λ, n] as the yield \nstress, consistency, and flow index,\n respectively"

        BinghamModel="Enter values for [μo, μf] as the yield \nstress and plastic viscosity, respectively"

        EllisModel="Enter values for [μo, μf, λ, n] as the zero \nshear viscosity, infinite viscosity, \nconsistency, and power law index, \nrespectively"

        SiskoModel="Enter values for [μf, λ, n] as the infinite \nviscosity, consistency and power law \nindex, respectively"

        PowerLawModel="Enter values for [λ, n] as the consistency \nand power law index, respectively"

        WilliamsonModel="Enter values for [μo, λ, n] as the zero \nshear viscosity, consistency, and power \nlaw index, respectively"

        CrossModel="Enter values for [μo, μf, λ, n] as the zero \nshear viscosity, infinite viscosity, \nconsistency, and power law index, \nrespectively"

        CassonModel="Enter values for [μo, μf] as the Casson \nyield stress and Casson viscosity, \nrespectively"

        PowellEyringModel="Enter values for [μo, μf, λ] as the zero \nshear viscosity, infinite viscosity, \n and consistency, \nrespectively"

        if model == "HerschelBulkley":
            msg=HerschelBulkleyModel

        elif model=="Carreau-Yasuda":
            msg=CarreauYasudaModel

        elif model=="Bingham":
            msg=BinghamModel

        elif model=="Ellis":
            msg=EllisModel

        elif model=="Sisko":
            msg=SiskoModel

        elif model=="Power-Law":
            msg=PowerLawModel

        elif model=="Williamson":
            msg=WilliamsonModel

        elif model=="Cross":
            msg=CrossModel

        elif model=="Casson":
            msg=CassonModel

        elif model=="PowellEyring":
            msg=PowellEyringModel

        self.pop= Toplevel(self.GUI)
        self.pop.geometry("300x100")
        self.pop.title("Parameters Window")
        Label(self.pop,text=msg, font=('none 11 bold'),justify=LEFT).place(x=5,y=5)

    def close_window(self):
        self.GUI.destroy()
        exit()

    def make_textmenu(self):
        global m
        m = Menu(self.GUI, tearoff=0)
        m.add_command(label="Cut")
        m.add_command(label="Copy")
        m.add_command(label="Paste")
        m.add_separator()
        m.add_command(label="Select all")

    def callback_select_all(self, event):
        self.GUI.after(50, lambda:event.widget.select_range(0, 'end'))

    def show_textmenu(self, event):
        self.e_widget = event.widget
        m.entryconfigure("Cut",command=lambda: self.e_widget.event_generate("<<Cut>>"))
        m.entryconfigure("Copy",command=lambda: self.e_widget.event_generate("<<Copy>>"))
        m.entryconfigure("Paste",command=lambda: self.e_widget.event_generate("<<Paste>>"))
        m.entryconfigure("Select all",command=lambda: self.e_widget.select_range(0, 'end'))
        m.tk.call("tk_popup", m, event.x_root, event.y_root)

    def osPath(self):

        self.shear_rate.clear() 
        self.viscosity.clear()
        self.cwd = os.getcwd()
        os.path.filename = filedialog.askopenfilename(
                                                    parent=self.GUI,initialdir=self.cwd,
                                                    title='Please select a directory',
                                                    filetypes=(("dat files", "*.dat"),("text files", "*.txt"),
                                                    ("csv files", "*.csv"),)
                                                   )

        if os.path.filename:
            self.upload_txtentry.delete(0,END)
            self.upload_txtentry.insert(0,os.path.filename)
            os.path.filename=self.upload_txtentry.get()
            


            try:
                for i in np.loadtxt(os.path.filename):
                    self.shear_rate.append(i[0])
                    self.viscosity.append(i[1])

            except:
                try:
                    for i in np.loadtxt(os.path.filename,skiprows=2):
                        self.shear_rate.append(i[0])
                        self.viscosity.append(i[1])
                except:
                    self.open_popup()

    def PlotPowerLaw(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = False
        try:
            self.n  = float(self.plaw_const.get("1.0",'end-1c'))
        except:
            self.n = False

        if self.λ==0 or self.n==0:self.default=True

        if self.default or any([self.λ, self.n]):
            try:
                self.FitPowerLawModel(self.x, self.y, self.λ, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitPowerLawModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotSisco(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μf = float(self.inf_vis.get("1.0",'end-1c'))
        except:
            self.μf = False
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = False
        try:
            self.n  = float(self.plaw_const.get("1.0",'end-1c'))
        except:
            self.n = False

        if any(i == 0 for i in [self.μf, self.λ, self.n]):self.default=True

        if self.default or any([self.μf, self.λ, self.n]):
            try:
                self.FitSiskoModel(self.x, self.y, self.μf, self.λ, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitSiskoModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotCross(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μo = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μo = False
        try:
            self.μf = float(self.inf_vis.get("1.0",'end-1c'))
        except:
            self.μf = False
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = False
        try:
            self.n  = float(self.pw_indx.get("1.0",'end-1c'))
        except:
            self.n = False

        if any(i == 0 for i in [self.μo, self.μf, self.λ, self.n]):self.default=True

        if self.default or any([self.μo, self.μf, self.λ, self.n]):
            try:
                self.FitCrossModel(self.x, self.y, self.μo, self.μf, self.λ, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitCrossModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotPowellEyring(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μo = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μo = False
        try:
            self.μf = float(self.inf_vis.get("1.0",'end-1c'))
        except:
            self.μf = False
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = False

        if any(i == 0 for i in [self.μo, self.μf, self.λ]):self.default=True

        if self.default or any([self.μo, self.μf, self.λ]):
            try:
                self.FitPowellEyringModel(self.x, self.y, self.μo, self.μf, self.λ)
            except:
                self.open_popup()
        else:
            try:
                self.FitPowellEyringModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotEllis(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μo = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μo = False
        try:
            self.μf = float(self.inf_vis.get("1.0",'end-1c'))
        except:
            self.μf = False
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = False
        try:
            self.n  = float(self.pw_indx.get("1.0",'end-1c'))
        except:
            self.n = False

        if any(i == 0 for i in [self.μo, self.μf, self.λ, self.n]):self.default=True

        if self.default or any([self.μo, self.μf, self.λ, self.n]):
            try:
                self.FitEllisModel(self.x, self.y, self.μo, self.μf, self.λ, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitEllisModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotWilliamson(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.λ = float(self.cons_indx.get("1.0",'end-1c'))
        except:
            self.λ = False
        try:
            self.n  = float(self.plaw_const.get("1.0",'end-1c'))
        except:
            self.n = False
        try:
            self.μf = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μf = False

        if any(i == 0 for i in [self.μf, self.λ, self.n]):self.default=True

        if self.default or any([self.μf, self.λ, self.n]):
            try:
                self.FitWilliamsonModel(self.x, self.y, self.μf, self.λ, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitWilliamsonModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotCarreauYasuda(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μo = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μo = None
        try:
            self.μf = float(self.inf_vis.get("1.0",'end-1c'))
        except:
            self.μf = None
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = None
        try:
            self.a  = float(self.trans.get("1.0",'end-1c'))
        except:
            self.a = None
        try:
            self.n  = float(self.pw_indx.get("1.0",'end-1c'))
        except:
            self.n = None

        if any(i == 0 for i in [self.μo, self.μf, self.λ, self.a, self.n]):self.default=True

        if self.default or any([self.μo, self.μf, self.λ, self.a, self.n]):
            try:
                self.FitCarreauYasudaModel(self.x, self.y, self.μo, self.μf, self.λ, self.a, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitCarreauYasudaModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotBingham(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μo = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μo = None
        try:
            self.μf = float(self.inf_vis.get("1.0",'end-1c'))
        except:
            self.μf = None

        if self.μo==0 or self.μf==0:self.default=True

        if self.default or any([self.μo, self.μf]):

            try:
                self.FitBinghamModel(self.x, self.y, self.μf, self.μo)
            except:
                self.open_popup()
        else:
            try:
                self.FitBinghamModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotHerschelBulkley(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.μo = float(self.zero_vis.get("1.0",'end-1c'))
        except:
            self.μo = None
        try:
            self.λ  = float(self.lamda.get("1.0",'end-1c'))
        except:
            self.λ = None
        try:
            self.n  = float(self.pw_indx.get("1.0",'end-1c'))
        except:
            self.n = None

        if self.μo==0 or self.λ==0 or self.n==0:self.default=True

        if self.default or any([self.μo, self.λ, self.n]):
            try:
                self.FitHerschelBulkleyModel(self.x, self.y, self.μo, self.λ, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitHerschelBulkleyModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotCasson(self):

        self.default=False

        if self.shear_rate and self.viscosity:
            self.x, self.y = np.array(self.shear_rate), np.array(self.viscosity)
        else:
            try:
                x  = self.data_x_axis.get("1.0",'end-2c').rstrip()
                y  = self.data_y_axis.get("1.0",'end-2c')
                self.x = np.array([float(i) for i in x.split()])
                self.y = np.array([float(i) for i in y.split()])
            except:
                self.open_popup()
                return
        try:
            self.k = float(self.cons_indx.get("1.0",'end-1c'))
        except:
            self.k = False
        try:
            self.n  = float(self.plaw_const.get("1.0",'end-1c'))
        except:
            self.n = False

        if self.k==0 or self.n==0:self.default=True

        if self.default or any([self.k, self.n]):
            try:
                self.FitCassonModel(self.x, self.y, self.k, self.n)
            except:
                self.open_popup()
        else:
            try:
                self.FitCassonModel(self.x, self.y)
            except:
                self.open_popup()

    def PlotData(self):

        try:
            self.pop.destroy()
        except:
            pass
            
        model= self.model.get()

        if model=="Power-Law":
            self.PlotPowerLaw()

        elif model=="Williamson":
            self.PlotWilliamson()

        elif model=="Cross":
            self.PlotCross()

        elif model=="Sisko":
            self.PlotSisco()

        elif model=="Carreau-Yasuda":
            self.PlotCarreauYasuda()

        elif model=="Ellis":
            self.PlotEllis()

        elif model=="HerschelBulkley":
            self.PlotHerschelBulkley()

        elif model=="Bingham":
            self.PlotBingham()

        elif model=="Casson":
            self.PlotCasson()

        elif model=="PowellEyring":
            self.PlotPowellEyring()

if __name__ == "__main__":

    GraphicalUserInterface()
