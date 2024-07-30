
# This file is your entry point:
# - add you Python files and folder inside this 'flows' folder
# - add your imports
# - just don't change the name of the function 'run()' nor this filename ('pybarsimtest.py')
#   and everything is gonna be ok.
#
# Remember: everything is gonna be ok in the end: if it's not ok, it's not the end.
# Alternatively, ask for help at https://github.com/deeplime-io/onecode/issues

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
# from IPython.display import HTML

from pybarsim import BarSim2D

from onecode import slider, image_output, video_output, Logger, checkbox


def run():
    # ## 1. Setup and run

    # Define the initial elevation and cell size (in m):

    # In[ ]:


    initial_elevation = np.linspace(1000., 900., 200)

    spacing = 100.


    # Define the run time (in yr) and the inflection points for the variations of sea level (in m):

    # In[ ]:


    run_time = slider('run time', 25000., min=10000, max=50000, step=1000)

    sea_level_curve = np.array([
        (0., 998.),
        (0.25*run_time, 985.),
        (0.5*run_time, 975.),
        (0.75*run_time, 985.),
        (run_time, 998.)
    ])


    # Define the inflection points for the variations of sediment supply (in m$^2$/yr):

    # In[ ]:


    sediment_supply_curve = np.array([
        (0., 25.),
        (0.25*run_time, 25.),
        (0.5*run_time, 25.),
        (0.75*run_time, 5.),
        (run_time, 5.)
    ])


    # Initialize a `BarSim2D` object and run the simulation:

    # <div class="alert alert-block alert-warning">
    # <b>&#9888;</b> This takes more time to run the first time because Numba needs to compile the Python code (around 15 s against less than 1 s for the following runs).
    # </div>

    # In[ ]:


    Logger.info('=== Starting simulation ===')
    barsim = BarSim2D(initial_elevation,
                      sea_level_curve,
                      sediment_supply_curve,
                      spacing=spacing,
                      max_wave_height_fair_weather=1.5,
                      allow_storms=checkbox('storms?', True),
                      start_with_storm=checkbox('start with storms?', False),
                      max_wave_height_storm=slider('max wave height', 6., min=0.1, max=25.),
                      tidal_amplitude=slider('tidal amp', 2., min=1, max=10, step=1),
                      min_tidal_area_for_transport=slider('tidal area', 100., min=10, max=500, step=10),
                      sediment_size=(5., 50., 125., 250.),
                      sediment_fraction=(0.25, 0.25, 0.25, 0.25),
                      initial_substratum=(100., (0.25, 0.25, 0.25, 0.25)),
                      erodibility=0.1,
                      washover_fraction=0.5,
                      tide_sand_fraction=0.3,
                      depth_factor_backbarrier=5.,
                      depth_factor_shoreface=10.,
                      local_factor_shoreface=1.5,
                      local_factor_backbarrier=1.,
                      fallout_rate_backbarrier=0.,
                      fallout_rate_shoreface=0.0002,
                      max_width_backbarrier=500.,
                      curve_preinterpolation=None,
                      seed=42)
    barsim.run(run_time, dt_fair_weather=15., dt_storm=1.)

    # `run` creates `sequence_`, a [xarray](https://docs.xarray.dev/en/stable/) dataset containing the values of sea level, sediment supply, elevation, stratigraphy, and facies in time:

    # In[ ]:


    Logger.info(barsim.sequence_)


    # We can visualize all the variables and their variation through time using [xarray's plotting functions](https://docs.xarray.dev/en/stable/user-guide/plotting.html?highlight=plotting):

    # In[ ]:


    barsim.sequence_['Sea level'].plot()
    plt.savefig(image_output('sea level', 'sea_level.png'))


    # In[ ]:


    barsim.sequence_['Elevation'].plot()
    plt.savefig(image_output('elevation', 'elevation.png'))


    # ## 2. Stratigraphy visualization

    # In `sequence_`, the stratigraphy is directly the final stratigraphy (i.e., it stores the remaining deposits after erosion), while the elevation corresponds to the true evolution of elevation through time. To visualize the final stratigraphy, `finalize` will update the elevation to account for erosion (new variable `Horizons`), and compute the mean grain size and the sorting term:

    # In[ ]:


    Logger.info('=== Finalizing erosion ===')
    barsim.finalize()


    # In[ ]:


    Logger.info(barsim.sequence_)


    # When the number of time steps gets too high, plotting takes longer and can become distorted. `subsample` creates `subsequence_`, a [xarray](https://docs.xarray.dev/en/stable/) dataset with a given number of time steps (here 20) merged together:

    # In[ ]:


    barsim.subsample(20)


    # In[ ]:


    Logger.info(barsim.subsequence_)


    # Calling `finalize` again specifically on `subsequence_` computes the mean grain size, the sorting term, and the major facies:

    # In[ ]:


    barsim.finalize(on='subsequence')


    # In[ ]:


    Logger.info(barsim.subsequence_)


    # Similarly to `sequence_`, we can visualize the values in time using [xarray's plotting functions](https://docs.xarray.dev/en/stable/user-guide/plotting.html?highlight=plotting):

    # In[ ]:

    Logger.info('=== Making plots ===')
    barsim.subsequence_['Mean grain size'].plot();
    plt.savefig(image_output('mean grain size', 'mean_grain_size.png'))


    # Or we can use the function `plot_subsequence` to plot the final stratigraphy in space:

    # In[ ]:


    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(barsim.subsequence_['X'][:-1],
                    barsim.subsequence_['Horizons'][0, :-1],
                    barsim.subsequence_['Horizons'][0, :-1].min(),
                    color='#d9d9d9')
    c = barsim.plot_subsequence(ax, var='Mean grain size')
    fig.colorbar(c[0], ax=ax, label=r'Mean grain size ($\mu$m)')
    ax.set(xlabel='x (m)', ylabel='z (m)');


    # In[ ]:


    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(barsim.subsequence_['X'][:-1],
                    barsim.subsequence_['Horizons'][0, :-1],
                    barsim.subsequence_['Horizons'][0, :-1].min(),
                    color='#d9d9d9')
    c = barsim.plot_subsequence(ax,
                                var='Major facies',
                                cmap='Set2',
                                norm=mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], 6))
    cbar = fig.colorbar(c[0], ax=ax, label=r'Major facies')
    cbar.set_ticks(ticks=[1, 2, 3, 4, 5, 6],
                   labels=barsim.subsequence_['Environment'].values[1:])
    ax.set(xlabel='x (m)', ylabel='z (m)');
    plt.savefig(image_output('major facies 2', 'major_facies2.png'))


    # We can also plot specific grain sizes or facies using the `idx` parameter:

    # In[ ]:


    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(barsim.subsequence_['X'][:-1],
                    barsim.subsequence_['Horizons'][0, :-1],
                    barsim.subsequence_['Horizons'][0, :-1].min(),
                    color='#d9d9d9')
    c = barsim.plot_subsequence(ax, var='Facies', idx=5, mask_zeros=False)
    fig.colorbar(c[0], ax=ax, label=r'Fraction of ' + str(barsim.subsequence_['Environment'][5].values))
    ax.set(xlabel='x (m)', ylabel='z (m)');
    plt.savefig(image_output('facies', 'facies.png'))


    # ## 3. Stratigraphy regridding

    # `subsequence_`'s grid is irregular and can be difficult to use in subsequent simulations. `regrid` reinterpolates BarSim's outputs on a regular grid:

    # In[ ]:


    Logger.info('=== Stratigraphy regridding ===')
    barsim.regrid(900., 1000., 0.5)


    # `regrid` creates `record_`, a [xarray](https://docs.xarray.dev/en/stable/) dataset containing the stratigraphy and facies in space:

    # In[ ]:


    barsim.record_


    # `finalize` computes once again the mean grain size, the sorting term, and the major facies:

    # In[ ]:


    barsim.finalize(on='record')


    # In[ ]:


    barsim.record_


    # We can visualize the resulting grid using [xarray's plotting functions](https://docs.xarray.dev/en/stable/user-guide/plotting.html?highlight=plotting):

    # In[ ]:


    Logger.info('=== Making plots ===')
    barsim.record_['Mean grain size'].plot(figsize=(12, 4))
    plt.savefig(image_output('mean grain size 2', 'mean_grain_size2.png'))


    # In[ ]:


    barsim.record_['Sorting term'].plot(figsize=(12, 4))
    plt.savefig(image_output('sorting term', 'sorting_term.png'))


    # In[ ]:


    fig, ax = plt.subplots(figsize=(12, 4))
    im = barsim.record_['Major facies'].where(barsim.record_['Major facies'] > 0, np.nan).plot(ax=ax,
                                                                                               cmap='Set2',
                                                                                               norm=mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 7),
                                                                                               add_colorbar=False)
    cbar = fig.colorbar(im, ax=ax, label=r'Major facies')
    cbar.set_ticks(ticks=[1, 2, 3, 4, 5, 6, 7],
                   labels=barsim.subsequence_['Environment'].values);
    plt.savefig(image_output('major facies', 'major_facies.png'))


    # ## 4. Process visualization

    # Let's create a simple animation of the variations of sea level and elevation through time using [matplotlib](https://matplotlib.org/stable/api/animation_api.html):

    # <div class="alert alert-block alert-warning">
    # <b>&#9888;</b> This takes some time to run.
    # </div>

    # In[ ]:


    fig, ax = plt.subplots(figsize=(12, 4))

    step = 15
    time = barsim.sequence_['Time'][::step]
    sea_level = barsim.sequence_['Sea level'][::step]
    elevation = barsim.sequence_['Elevation'][::step, :-1].copy()
    x = barsim.sequence_['X'].values[:-1]

    def update(i):
        label_time.set_text(str(round(int(time[i]), -min(2, int(np.log10(time[i] + 1e-8))))) + ' yr')
        path = fill_sea.get_paths()[0]
        path.vertices[len(elevation[i]) + 2:-1, 1] = sea_level[i]
        path = fill_subsurface.get_paths()[0]
        path.vertices[len(elevation[i]) + 2:-1, 1] = elevation[i][::-1]
        k = 0
        for j in range(i):
            if j%15 == 0:
                elevation_prev = elevation[j].to_numpy()
                elevation_prev[elevation_prev > elevation[i]] = elevation[i][elevation_prev > elevation[i]]
                lines_elevation_prev[k].set_data((x, elevation_prev))
                k += 1
        line_elevation.set_ydata(elevation[i])
        return label_time, fill_sea, fill_subsurface, line_elevation, lines_elevation_prev

    ax.annotate('Time:', (0.85, 0.92), xycoords='axes fraction')
    label_time = ax.annotate(str(round(int(time[0]), -min(2, int(np.log10(time[0] + 1e-8))))) + ' yr', (0.965, 0.92), ha='right', xycoords='axes fraction')
    fill_sea = ax.fill_between(x, elevation.min(), sea_level[0], edgecolor='#6baed6', facecolor='#c6dbef', zorder=0)
    fill_subsurface = ax.fill_between(x, elevation.min(), elevation[0], color='#fff7bc', zorder=1)
    lines_elevation_prev = [ax.plot([], [], c='0.5', lw=0.5, zorder=2)[0] for i in range(len(elevation[::15]))]
    line_elevation, = ax.plot(x, elevation[0], c='k', lw=1.5, zorder=3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(elevation.min(), elevation.max() + 20.)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')

    plt.close()

    Logger.info('=== Making video ===')
    ani = animation.FuncAnimation(fig, update, len(sea_level), interval=100)
    # HTML(ani.to_jshtml())
    ani.save(video_output('sim', 'simulation.mp4'))

    Logger.info('=== Done! ===')
