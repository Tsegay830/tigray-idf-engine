# ------------------------------------------
# TAB 1: REGIONAL LINEAR & LOGARITHMIC IDF CURVES
# ------------------------------------------
with tab1:
    st.subheader(f"⚡ Intensity-Duration-Frequency (IDF) Curves: {selected_town}")
    st.caption("Dual-visualization (Linear vs. Logarithmic Duration Axis) sampled at 15-minute intervals for short-duration high-intensity convective storms.")
    
    # 1. Fine-grained durations: 15-min intervals up to 180 min, then key macro durations up to 24h
    short_durations = np.arange(15, 195, 15)  # 15, 30, 45, 60, ..., 180 min
    macro_durations = np.array([240, 360, 540, 720, 1080, 1440]) # 4h, 6h, 9h, 12h, 18h, 24h
    durations_min = np.unique(np.concatenate([short_durations, macro_durations]))
    
    return_periods = [2, 5, 10, 25, 50, 100]
    
    # Generate perturbed empirical IDF dataset
    np.random.seed(abs(hash(selected_town)) % 999)
    idf_data = {}
    for T in return_periods:
        base_i = (480 * (T ** 0.22)) / (durations_min + 18)
        messed_noise = np.random.normal(0, 2.5, size=len(durations_min))
        intensity = np.maximum(base_i + messed_noise, 2.0)
        idf_data[T] = intensity

    # Figure setup: 2 Stacked Subplots (Linear Top, Logarithmic Bottom)
    fig_idf, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    
    # ------------------------------------------
    # GRAPH 1: LINEAR PLOT (Focused on 15-min Intervals)
    # ------------------------------------------
    for T in return_periods:
        ax_lin.plot(durations_min, idf_data[T], label=f"T = {T} Years", linewidth=1.8, marker='o', markersize=4, linestyle='--')
    
    ax_lin.set_xlim(0, 185) # Focused zoom on short-duration / high-intensity zone (0 - 180 min)
    ax_lin.set_xticks(np.arange(15, 195, 15)) # 15-minute ticks on x-axis
    
    # 2 mm/hr interval ticks on y-axis
    max_val = max([max(v) for v in idf_data.values()])
    ax_lin.set_yticks(np.arange(0, np.ceil(max_val) + 4, 2))
    
    ax_lin.set_xlabel("Storm Duration (minutes) [Linear Scale: 15-min Ticks]", fontweight="bold")
    ax_lin.set_ylabel("Rainfall Intensity (mm/hr) [2 mm/hr Ticks]", fontweight="bold")
    ax_lin.set_title(f"1. Linear IDF Plot (High-Intensity Zone 15–180 min) - {selected_town}", fontweight="bold")
    ax_lin.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_lin.legend(title="Return Period", loc="upper right")

    # ------------------------------------------
    # GRAPH 2: SEMILOGARITHMIC PLOT (Full 15 min to 24 Hours)
    # ------------------------------------------
    for T in return_periods:
        ax_log.plot(durations_min, idf_data[T], label=f"T = {T} Years", linewidth=1.8, marker='s', markersize=4)
    
    ax_log.set_xscale('log') # Logarithmic duration scale
    ax_log.set_yticks(np.arange(0, np.ceil(max_val) + 4, 2)) # 2 mm/hr ticks maintained
    
    ax_log.set_xlabel("Storm Duration (minutes) [Logarithmic Scale: 15 min to 1440 min]", fontweight="bold")
    ax_log.set_ylabel("Rainfall Intensity (mm/hr) [2 mm/hr Ticks]", fontweight="bold")
    ax_log.set_title(f"2. Semilogarithmic IDF Plot (Extended Duration Spectrum 15 min – 24 hrs) - {selected_town}", fontweight="bold")
    ax_log.grid(True, which="both", linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig_idf)
