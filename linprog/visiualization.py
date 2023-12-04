import intvalpy as ip
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def simplex_method_visualization(A, b, vertex_bypass_info):
    dim = A.shape[1]

    if dim == 2:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, title='Problem feasible region')
        vertices = ip.lineqs(A, b,show=False)
        x, y = vertices[:, 0], vertices[:, 1]
        ax.fill(x, y, 'r', alpha=0.3)
        ax.scatter(x, y,  s=100, marker='o')
        route = [v[1][:dim] for v in vertex_bypass_info]
        
        for i, v in enumerate(route):
            ax.scatter(v[0], v[1])
            vert_text = f'vertex {i + 1}:\nobj. func: {vertex_bypass_info[i][0]} \ncoord: {v[0], v[1]}'
            ax.annotate(vert_text, (v[0], v[1]), (v[0] + 0.1, v[1] + 0.1), fontsize=20)

    elif dim == 3:
        fig = plt.figure(figsize=(12, 12))
        ax = Axes3D(fig)
        fig.add_axes(ax)
        ax.set_xlim3d(0, 5)
        ax.set_ylim3d(0, 5)
        ax.set_zlim3d(0, 5)

        vertices = ip.lineqs3D(A, b, show=False)

        for v in vertices:
            face = Poly3DCollection([v])
            face.set_alpha(0.3)
            face.set_edgecolor('k')
            ax.add_collection3d(face)

        route = [v[1][:dim] for v in vertex_bypass_info]
        
        for i, v in enumerate(route):
            ax.scatter(v[0], v[1], v[2], s=1000)
            vert_text = f'vertex {i + 1}:\n obj. func: {vertex_bypass_info[i][0]} \ncoord: {v[0], v[1], v[2]}'
            ax.text3D(v[0] + 0.1, v[1] + 0.1, v[2] + 0.1, vert_text)
    
    plt.show()