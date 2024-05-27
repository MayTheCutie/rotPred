def i_p_scatter(kepler_inference, kepler_inference2=None, conf=None, dir='../imgs'):
    """
    scatter period vs inclination
    """
    print("max period: ", kepler_inference['predicted period'].max())
    fig, axes = plt.subplots()
    axes.scatter(kepler_inference['predicted period'], kepler_inference['predicted inclination'], label='all data')
    if kepler_inference2 is not None:
        axes.scatter(kepler_inference2['predicted period'], kepler_inference2['predicted inclination'], label='h')
    if conf is not None:
        for c in conf:
            reduced = kepler_inference[kepler_inference['confidence'] > c]
            if len(reduced > 10):
                plt.scatter(reduced['planet_Prot'], reduced['predicted inclination'],
                            label=f'conf > {c} {len(reduced)} points', alpha=0.4)
    plt.xlabel('period')
    plt.ylabel('inclination')
    # plt.xlim(0,60)
    plt.title('inc vs p - confidence')
    plt.legend(loc='upper right')
    plt.savefig(f'{dir}/inc_p_scatter_conf.png')
    plt.show()
    plt.close('all')

def dist_test(df1, df2):
    p_values = []
    confidence_values = np.arange(0.85,1,0.01)
    teff_values = np.arange(4000,7000,100)
    for confidence in confidence_values:
        for teff in teff_values:
            print(confidence, teff)
            subset1 = df1[(df1['inclination confidence'] >= confidence) & (df1['Teff'] >= teff)][
                'predicted inclination']
            subset2 = df2['predicted inclination']
            if len(subset1) > 100 and len(subset2) > 100:
                _, p_value = ks_2samp(subset1, subset2)
                p_values.append((confidence, teff, p_value*np.log(len(subset1))))
            else:
                p_values.append((confidence, teff, 0))


    # Convert p_values to a DataFrame
    result_df = pd.DataFrame(p_values, columns=['inclination confidence', 'Teff', 'p-value'])
    # Plot the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(teff_values, confidence_values,)
    Z = np.array(result_df['p-value']).reshape(len(confidence_values), -1)

    max_idx = np.argmax(Z)


    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, alpha=0.8, antialiased=True)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('K-S p-value')

    # ax.scatter(result_df['inclination confidence'], result_df['Teff'], result_df['p-value'], c='r', marker='o')

    ax.set_ylabel('Confidence', fontsize=14)
    ax.set_xlabel('Teff', fontsize=14)
    ax.set_zlabel('log(K-S p-value)', fontsize=14)

    plt.title('Kolmogorov-Smirnov Test Results')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    plt.savefig(r"C:\Users\ilaym\Desktop\kepler\acf\analyze\imgs\ks_test.png")
    plt.show()

    # ax.scatter(result_df['inclination confidence'], result_df['Teff'], result_df['p-value'], c='r', marker='o')
    # ax.set_xlabel('Inclination Confidence')
    # ax.set_ylabel('Teff')
    # ax.set_zlabel('K-S p-value')
    #
    # plt.title('Kolmogorov-Smirnov Test Results')
    # plt.show()

def age_analysis(kepler_inference, age_df, refs, refs_names, age_vals=[1]):
    age_df = age_df[(age_df['E_Age'] <= 1) & (age_df['e_Age'].abs() <= 1)]
    kepler_inference['age_model'] = calc_gyro_age_gyears(kepler_inference['predicted period'],
                                                  kepler_inference['Teff'])
    merged_df = kepler_inference.merge(age_df, on='KID')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
    p_err_model = np.vstack([merged_df['period model error lower'].values[None],
                             merged_df['period model error lower'].values[None]])
    model_std = np.std(merged_df['predicted period'])
    plt.scatter(merged_df['Age'], merged_df['age_model'])
    for name, ref in zip(refs_names, refs):
        suffix = '_' + name
        merged_df = merged_df.merge(ref, on='KID', suffixes=(None, suffix))
        merged_df[f'age_{name}'] = calc_gyro_age_gyears(merged_df[f'Prot_{name}'], merged_df['Teff'])
        plt.scatter(merged_df['Age'], merged_df[f'age_{name}'])
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def calc_gyro_age_gyears(p, Teff):
    a = 0.77
    b = 0.553
    c = 0.472
    n = 0.519
    B_V = B_V_from_T(Teff)
    log_t = (1/n) * (np.log10(p) - np.log10(a) - b*np.log10(B_V - c))
    return 10**(log_t)*1e-3


def B_V_from_T(T):
    a = 0.8464 * T
    b = 2.1344 * T - 4600 * 1.84
    c = 1.054 * T - 4600 * 2.32

    discriminant = b ** 2 - 4 * a * c

    x_positive = (-b + np.sqrt(discriminant)) / (2 * a)
    return x_positive

def T_from_B_V(B_V):
    return 4600*(1/(0.92*B_V+1.7)+1/(0.92*B_V + 0.62))


def ssl_vs_finetune(finetune_df, ssl_df):
    merged_df = pd.merge(finetune_df, ssl_df, on='KID', suffixes=['_ssl_finetune', '_ssl'])
    p_diff = np.abs(merged_df['predicted period_ssl_finetune'] - merged_df['predicted period_ssl'])
    avg_diff = p_diff.mean()
    std_diff = p_diff.std()
    plt.scatter(merged_df['predicted period_ssl_finetune'], merged_df['predicted period_ssl'], alpha=0.5)
    plt.plot(merged_df['predicted period_ssl_finetune'], merged_df['predicted period_ssl_finetune'], color='r')
    plt.xlabel('Predicted Period w/ finetune (Days)')
    plt.ylabel('Predicted Period w/o finetune (Days)')
    plt.savefig('../imgs/ssl_vs_finetune.png')
    plt.close()
    return avg_diff, std_diff

def classification_inference():
    kepler_inference = read_csv_folder('../inference/astroconf_cls_exp7_ssl_finetuned',
                                       filter_thresh=2, att='inclination class',
                                       scale=False, calc_errors=False)
    print("number of samples: ", len(kepler_inference))
    num_samples = []
    threshsholds = np.arange(0.1, 0.2, 0.01)
    for t in threshsholds:
        kepler_inference_t = kepler_inference[kepler_inference['predicted inclination probability'].apply
        (lambda x: max(x) > t)]
        plt.hist(kepler_inference_t['predicted inclination class'], histtype='step', bins=10, density=True)
        plt.show()
        num_samples.append(len(kepler_inference_t))
    plt.scatter(threshsholds, num_samples)
    plt.show()