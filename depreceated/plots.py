def plot_spots(data_dir, samples_list, dur=1000, t_s=0, t_e=1000):
    spots_dir = os.path.join(data_dir, "spots")
    df = pd.read_csv(os.path.join(data_dir, "simulation_properties.csv"))
    for i in samples_list:
        print(i)
        idx = remove_leading_zeros(i)
        spot_props = pd.read_parquet(os.path.join(spots_dir, f"spots_{i}.pqt"))
        star_props = df.iloc[idx]
        lc = Spots(
            spot_props,
            incl=star_props["Inclination"],
            period=star_props["Period"],
            diffrot_shear=star_props["Shear"],
            alpha_med=np.sqrt(star_props["Activity Rate"])*3e-4,
            decay_timescale=star_props["Decay Time"],
            dur=dur
        )
        time = np.arange(t_s, t_e, 1)
        flux = 1 + lc.calc(time)
        fig,axes = plt.subplots(2,1,figsize=(10,10))
        axes[1].plot(time,flux)
        lc.plot_butterfly(fig, axes[0])
        axes[0].set_title(f"Period: {star_props['Period']}, Inclination: {star_props['Inclination']}, decay time: {star_props['Decay Time']}, activity rate: {star_props['Activity Rate']}")
        plt.savefig(f"{data_dir}/plots/{idx}.png")
