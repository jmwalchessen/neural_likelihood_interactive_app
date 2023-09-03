import os
import webbrowser
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import io
from io import BytesIO
import pickle
import tensorflow
from tensorflow import keras
from keras.models import model_from_json
from flask import Flask
from flask import request, Response, render_template
from gp_exact_likelihood_with_parallelization import *
from gp_neural_likelihood import *
import base64

app = Flask(__name__)

@app.route('/')
def layout():
    return render_template('layout.html')

@app.route('/julia/')
def aboutme():
    return render_template('aboutme.html')

def results(data, ll_data, psi_data, calibrated_psi_data):

    return render_template("gp.html",  data = data.decode('utf8'), ll_data = ll_data.decode('utf8'),
                               psi_data = psi_data.decode('utf8'), calibrated_psi_data = calibrated_psi_data.decode('utf8'))



    return '''
           <form method="POST">
               <div><label>seed: <input type="number" name="seed" step = "1" min = "0" defaultValue="1"></label></div>
               <div><label>length scale: <input type="number" name="lengthscale" min = "0" max = "2" step = "0.01"></label></div>
               <div><label>variance: <input type="number" name="variance" min="0" max = "2" step = "0.01"></label></div>
               <input type="submit" value="Submit">
           </form>'''

    

@app.route('/gp/', methods=['GET', 'POST'])
def gp():
    # handle the POST request
    if request.method == 'POST':
        
        seed = int(request.form.get('seed'))
        lengthscale = float(request.form.get('lengthscale'))
        variance = float(request.form.get('variance'))
        n = 25
        
        C = 5.99
        number_of_replicates = 1
        minX = -10
        minY = -10
        maxX = 10
        maxY = 10
        y_matrix = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                             number_of_replicates, seed)
        y_matrix = y_matrix.reshape((n**2))
        y_image = np.zeros((n, n))
        for i in range(0, n):

            y_image[i,:] = (y_matrix[(i*n):((i + 1)*int(n))]).reshape((n,))
        
        possible_sigmas = [.05*i for i in range(1, 41)]
        possible_lengthscales = [.05*i for i in range(1,41)]
        psi_field = produce_psi_field(possible_lengthscales, possible_sigmas, y_image, n)
        psi_confidence_region = produce_neural_likelihood_confidence_region(psi_field,
                                                                            possible_lengthscales,
                                                                            possible_sigmas, C)
        
        psi_polygon_figure = patch_polygon(psi_confidence_region, facecolor = "none",
                                       edgecolor = "black", linewidth = 5)
        calibrated_psi_field = produce_calibrated_psi_field(possible_lengthscales, possible_sigmas,
                                                            y_image, n)
        calibrated_psi_confidence_region = produce_neural_likelihood_confidence_region(
            calibrated_psi_field,
            possible_lengthscales,
            possible_sigmas, C)
        calibrated_psi_polygon_figure = patch_polygon(calibrated_psi_confidence_region,
                                                         facecolor = "none",
                                                         edgecolor = "black",
                                                         linewidth = 5)
                                                                                       
        ll_field = np.transpose(execute_parallelized_ll_field(y_matrix, possible_sigmas,
                                                              possible_lengthscales, minX,
                                                              maxX, minY, maxY, n))
        ll_confidence_region = produce_exact_likelihood_confidence_region(ll_field, possible_lengthscales,
                                                                          possible_sigmas, C)
        
        ll_polygon_figure = patch_polygon(ll_confidence_region, facecolor = "none",
                                       edgecolor = "black", linewidth = 5)

        matplotlib.use('Agg')
        figfile = BytesIO()
        fig, ax = plt.subplots(figsize = (10,11))
        pixel_plot = ax.imshow(
                y_image, interpolation='nearest', extent = (-10,10,-10,10), cmap = 'seismic')
        ticks = [-10, -5, 0, 5, 10]
        plt.xticks(ticks, fontsize= 20)
        plt.yticks(ticks, fontsize = 20)
        m0=round(np.quantile(y_image.flatten(), .01), 1)           # colorbar min value
        m1=round(np.quantile(y_image.flatten(), .99), 1)           # colorbar max value
        # to get ticks
        ticks = [m0, 0, m1]
        cbar = plt.colorbar(pixel_plot, ticks = ticks)
        # get label
        labels = [str(m0), '0', str(m1)]
        cbar.set_ticklabels(labels, fontsize = 20)
        fig.savefig(figfile, format='png')
        data = base64.encodebytes(figfile.getvalue())
        plt.close()
        
        matplotlib.use('agg')
        psifile = BytesIO()
        constant = 10
        fig, ax = plt.subplots()
        x = np.linspace(.05, 2, 40)
        y = np.linspace(.05, 2, 40)
        X, Y = np.meshgrid(x, y)
        Z = np.log(psi_field)
        Z = Z.reshape((40, 40))
        max_indices = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
        max_lengthscale = possible_lengthscales[max_indices[1]]
        max_variance = possible_sigmas[max_indices[0]]
        lower_bound = (np.amax(Z) - constant)
        color_levels = [np.amin(Z)] + [lower_bound + .1*i for i in range(0, 100)] + [np.amax(Z)+.01]
        cp = ax.contourf(X, Y, Z, vmin = lower_bound, levels = color_levels)
        ax.add_patch(psi_polygon_figure)
        ax.scatter(lengthscale, variance, s = 400, marker = "*", c = "black")
        ax.scatter(max_lengthscale, max_variance, s = 400, marker = "o", c= "red")
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='True',
                          markerfacecolor='black', markersize=20), 
                          Line2D([0], [0], marker='o', color='w', label='Estimate',
                          markerfacecolor='red', markersize=15), Line2D([0], [0], marker='_',
                          color='black', label='95% CR',
                          markerfacecolor='none', markersize=20, linewidth = 8)]
        ax.legend(handles = legend_elements, facecolor='white', framealpha=1, fontsize="10")
        cbar = fig.colorbar(cp)
        m1=round(np.max(Z), 1)          
        m0=round((np.max(Z)-10),1)
        ticks = np.linspace(m0, m1, 7)          
        # get label
        labels = [str(round(elem,1)) for elem in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels, fontsize = 10)
        ax.set_xlim(0.05, 2)
        ax.set_ylim(0.05, 2)
        plt.xlabel("length scale")
        plt.ylabel("variance")
        fig.savefig(psifile, format = "png")
        psi_data = base64.encodebytes(psifile.getvalue())
        plt.close()
        
        matplotlib.use('agg')
        calibrated_psifile = BytesIO()
        constant = 10
        fig, ax = plt.subplots()
        x = np.linspace(.05, 2, 40)
        y = np.linspace(.05, 2, 40)
        X, Y = np.meshgrid(x, y)
        Z = np.log(calibrated_psi_field)
        Z = Z.reshape((40, 40))
        max_indices = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
        max_lengthscale = possible_lengthscales[max_indices[1]]
        max_variance = possible_sigmas[max_indices[0]]
        lower_bound = (np.amax(Z) - constant)
        color_levels = [np.amin(Z)] + [lower_bound + .1*i for i in range(0, 100)] + [np.amax(Z)+.01]
        cp = ax.contourf(X, Y, Z, vmin = lower_bound, levels = color_levels)
        ax.add_patch(calibrated_psi_polygon_figure)
        ax.scatter(lengthscale, variance, s = 400, marker = "*", c = "black")
        ax.scatter(max_lengthscale, max_variance, s = 400, marker = "o", c= "red")
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='True',
                          markerfacecolor='black', markersize=20), 
                          Line2D([0], [0], marker='o', color='w', label='Estimate',
                          markerfacecolor='red', markersize=15), Line2D([0], [0], marker='_',
                          color='black', label='95% CR',
                          markerfacecolor='none', markersize=20, linewidth = 8)]
        ax.legend(handles = legend_elements, facecolor='white', framealpha=1, fontsize="10")
        cbar = fig.colorbar(cp)
        m1=round(np.max(Z), 1)          
        m0=round((np.max(Z)-10),1)
        ticks = np.linspace(m0, m1, 7)          
        # get label
        labels = [str(round(elem,1)) for elem in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels, fontsize = 10)
        ax.set_xlim(0.05, 2)
        ax.set_ylim(0.05, 2)
        plt.xlabel("length scale")
        plt.ylabel("variance")
        fig.savefig(calibrated_psifile, format = "png")
        calibrated_psi_data = base64.encodebytes(calibrated_psifile.getvalue())
        plt.close()
        
        matplotlib.use('agg')
        llfile = BytesIO()
        constant = 10
        fig, ax = plt.subplots()
        x = np.linspace(.05, 2, 40)
        y = np.linspace(.05, 2, 40)
        X, Y = np.meshgrid(x, y)
        Z = ll_field
        Z = Z.reshape((40, 40))
        max_indices = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
        max_lengthscale = possible_lengthscales[max_indices[1]]
        max_variance = possible_sigmas[max_indices[0]]
        lower_bound = (np.amax(Z) - constant)
        color_levels = [np.amin(Z)] + [lower_bound + .1*i for i in range(0, 100)] + [np.amax(Z)+.01]
        cp = ax.contourf(X, Y, Z, vmin = lower_bound, levels = color_levels)
        ax.add_patch(ll_polygon_figure)
        ax.scatter(lengthscale, variance, s = 400, marker = "*", c = "black")
        ax.scatter(max_lengthscale, max_variance, s = 400, marker = "o", c= "red")
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='True',
                          markerfacecolor='black', markersize=20), 
                          Line2D([0], [0], marker='o', color='w', label='Estimate',
                          markerfacecolor='red', markersize=15), Line2D([0], [0], marker='_',
                          color='black', label='95% CR',
                          markerfacecolor='none', markersize=20, linewidth = 8)]
        ax.legend(handles = legend_elements, facecolor='white', framealpha=1, fontsize="10")
        cbar = fig.colorbar(cp)
        m1=round(np.quantile(Z, .99), 1)          
        m0=round((np.max(Z)-constant),1)
        ticks = np.linspace(m0, m1, 7)          
        # get label
        labels = [str(round(elem,1)) for elem in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels, fontsize = 10)
        ax.set_xlim(0.05, 2)
        ax.set_ylim(0.05, 2)
        plt.xlabel("length scale")
        plt.ylabel("variance")
        fig.savefig(llfile, format = "png")
        ll_data = base64.encodebytes(llfile.getvalue())
        plt.close()

        #results(data, ll_data, psi_data, calibrated_psi_data)

        return render_template("gp.html",  data = data.decode('utf8'), ll_data = ll_data.decode('utf8'),
                               psi_data = psi_data.decode('utf8'), calibrated_psi_data = calibrated_psi_data.decode('utf8'))
        
        
    
        
        

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>Seed: <input type="number" name="seed" step = "1" min = "0"></label></div>
               <div><label>lengthscale: <input type="number" name="lengthscale" min = "0" step = "0.1"></label></div>
               <div><label>variance: <input type="number" name="variance" min="0" step = "0.1"></label></div>
               <input type="submit" value="Submit">
           </form>'''

def main():
    
    # The reloader has not yet run - open the browser
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8000/')

    # Otherwise, continue as normal
    app.run(host="127.0.0.1", port=8000)
    

#app.add_url_rule('/', 'get_data', get_data,
#            methods=['GET', 'POST'])

if __name__ == '__main__':

    main()