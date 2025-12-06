import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import matplotlib.pyplot as plt
    import pandas as pd
    from pypdf import PdfWriter
    import io, base64
    return alt, mo, np, pd


@app.cell
def _(mo):
    f0 = mo.ui.slider(start=-5000, stop=5000, step=1, value=0, label="")
    phi = mo.ui.slider(start=-180, stop=180, step=.1, value=0, label="")
    logR = mo.ui.slider(start=0, stop=3.0, step=0.01, value=1.5, label="")
    noise = mo.ui.slider(start=0, stop=1, step=0.01, value=0.1, label="")
    samp = mo.ui.slider(start=0, stop=100, step=1, value=50, label="")
    seed = mo.ui.slider(start=0, stop=5000, step=1, value=2500, label="")
    mode = mo.ui.radio(
        options={"Uniform Sampling": "US", "Non-Uniform Sampling": "NUS"},
        value="Uniform Sampling",
        label="**Sampling**"
    )
    
    return f0, logR, mode, noise, phi, samp, seed


@app.cell
def _():
    # html = ''
    # if save_btn.value:   # True after click
    #     chart.save("charts.pdf")
    #     buf = io.BytesIO()
    #     chart.save(buf, format="pdf")
    #     pdf_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    #     html = f'''
    #     <a download="charts.pdf"
    #        href="data:application/pdf;base64,{pdf_b64}">
    #        ⬇️ Download
    #     </a>
    #     '''


    #     # marimo 0.14 has mo.md; it renders HTML links fine
    #     mo.md(html)
    #     #mo.ui.download.from_path("charts.pdf", label="Download PDF")  # if available in your marimo version
    return


@app.cell
def _(R, f0, logR, mo, mode, noise, phi, samp, seed):
    mo.md(
        f"""
    **Controls** \
            \n {f0} $f_0$ = **{f0.value:.4g}** [Frequency] \
            \n {logR} $R$ = **{R:.4g}** [Relaxation] \
            \n {noise} $\sigma$ = **{noise.value:.4g}** [Noise] \
            \n {phi} $\phi$ = **{phi.value:.4g}** [Phase] \
            \n {samp} $\%$ = **{samp.value}** [Sampling Percentage]\
            \n {seed} $s$ = **{seed.value:.4g}** [Random Seed] \
            \n {mode} \
            \n \
    """
    )
    return


@app.cell
def _(chart):
    chart
    return


@app.cell
def _(f0, logR, mode, noise, np, phi, samp, seed):
    def complex_signal(f0, fs, N, A=1.0, phi=0.0, noise=0):
        """
        Complex exponential at one frequency:
            x[n] = A * exp(j*(2π f0 n/fs + phi)),  n = 0..N-1
        """
        n = np.arange(N, dtype=np.float64)
        noise_real = np.random.randn(N)*noise
        noise_imag = np.random.randn(N)*noise
        nc = noise_real +1j*noise_imag
        return A * np.exp(1j * (2*np.pi*f0*n/fs + phi)) + nc

    def NUS_Signal(x, N, rng, s):
                    # optional seed
        k = int(round(((s/100)) * N))                   # number of ones
        xx = np.zeros(N, dtype=np.int8)
        xx[rng.choice(N, size=k, replace=False)] = 1
        return x * xx


    rng = np.random.default_rng(seed.value)
    # example
    fs = 10000         # sample rate (Hz)
    #f0 = 10         # tone frequency (Hz)

    N  = 1000         # samples (1 second)
    n = np.arange(N)
    x = complex_signal(f0.value, fs, N, A=1.0, phi=phi.value*np.pi/(180), noise=noise.value)

    if mode.value == "NUS":
        x_s = NUS_Signal(x, N, rng, samp.value)

    else:
        x_s = x


    R = 10 ** logR.value
    x_relax = x_s * np.exp(-(n/N)*R)
    x_win = np.concat((x_relax * np.cos(np.pi*n/(2*N)), np.zeros(N)))

    N_zf = 2000
    n_zf = np.arange(N_zf)

    return N, R, fs, n_zf, x_win


@app.cell
def _(N, fs, np, x_win):
    X    = np.fft.fft(x_win)# * win)
    freqs = np.fft.fftfreq(2*N, d=1/(fs))
    return X, freqs


@app.cell
def _(X, alt, freqs, fs, mode, n_zf, np, pd, x_win):
    df = pd.DataFrame({
        "t": n_zf / fs,
        "freqs": freqs,
        "real": np.real(x_win),
        "imag": np.imag(x_win),
        "fft_real": np.real(X),
        "fft_imag": np.imag(X),
    })

    # (optional) Altair rows cap; disable if you have lots of samples
    alt.data_transformers.disable_max_rows()

    # 1) Real & Imag over time (overlay)
    df_long = df.melt(id_vars="t", value_vars=["real"],#, "imag"],
                      var_name="component", value_name="value")

    df_long_fft = df.melt(id_vars="freqs", value_vars=["fft_real"],# "fft_imag"],
                      var_name="component", value_name="value")


    if mode.value == "NUS":
        chart_ri = (
            alt.Chart(df_long).mark_point()
            .encode(
                x=alt.X("t:Q", title="Time (s)"),
                y=alt.Y("value:Q", title="Amplitude"),
                size=alt.value(4),
                #color=alt.Color("component:N", title="Component")
            )
            .properties(title='Signal (Real)', width=800, height=200)
            .interactive()
        )

    else:
        chart_ri = (
            alt.Chart(df_long).mark_line()
            .encode(
                x=alt.X("t:Q", title="Time (s)"),
                y=alt.Y("value:Q", title="Amplitude"),

                #color=alt.Color("component:N", title="Component")
            )
            .properties(title='Signal (Real)', width=800, height=200)
            .interactive()
        )


    chart_fft = (
        alt.Chart(df_long_fft).mark_line()
        .encode(
            x=alt.X("freqs:Q", title="Freq (Hz)"),
            y=alt.Y("value:Q", title="Amplitude"),
            #color=alt.Color("component:N", title="Component")
        )
        .properties(title='Fourier Transform (Real)', width=800, height=200)
        .interactive()
    )

    chart = chart_fft
    chart = ( chart_ri & chart_fft)

    return (chart,)


if __name__ == "__main__":
    app.run()
