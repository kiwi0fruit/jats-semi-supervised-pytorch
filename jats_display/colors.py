from matplotlib.colors import to_rgb

_types_colors = (
    'bright red', 'tangerine', 'dandelion', 'apple green',  # 'bright yellow'
    'magenta', 'purple', 'blue', 'teal blue',
    'grass green', 'gold', 'dark orange', 'blood red',
    'aqua blue', 'azure', 'bright violet', 'bright pink',  # 'bright purple'
)
types_color_names = (
    'red', 'orange', 'yellow', 'lime',
    'magenta', 'purple', 'blue', 'teal',
    'green', 'mustard', 'brown', 'crimson',
    'cyan', 'azure', 'violet', 'pink',
)
if len(_types_colors) != len(types_color_names):
    raise RuntimeError()
types_colors = tuple(to_rgb(f'xkcd:{s}') for s in _types_colors)


def plot():
    from IPython.display import HTML
    from kiwi_bugfix_typechecker.ipython import display
    import matplotlibhelper as mh
    mh.ready(font_size=12, ext='svg', hide=True)  # magic='agg',
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatch

    jats_names = {
        1: 'ENT/-ir, ILE, ENTP, NeT', 2: 'INT/ir, LII, INTx, TiN',
        3: 'ISF/-er, SEI, ISFx, SiF', 4: 'ESF/er, ESE, ESFJ, FeS',

        5: 'EST/-ir, SLE, ESTP, SeT', 6: 'IST/ir, LSI, ISTx, TiS',
        7: 'INF/-er, IEI, INFx, NiF', 8: 'ENF/er, EIE, ENFJ, FeN',

        9: 'ESF/-ir, SEE, ESFP, SeF', 10: 'ISF/ir, ESI, ISFx, FiS',
        11: 'INT/-er, ILI, INTx, NiT', 12: 'ENT/er, LIE, ENTJ, TeN',

        13: 'ENF/-ir, IEE, ENFP, NeF', 14: 'INF/ir, EII, INFx, FiN',
        15: 'IST/-er, SLI, ISTx, SiT', 16: 'EST/er, LSE, ESTJ, TeS'}

    fig = plt.figure(figsize=[4.3 + 0.5, 6])
    ax = fig.add_axes([0, 0, 1, 1])

    for j, (color_name, color) in enumerate(zip(reversed(types_color_names), reversed(types_colors))):
        x0 = 0.65
        r1 = mpatch.Rectangle((0, j), x0, 1, color=color)
        ax.text(x0, j + .5, f'  {color_name}', va='center')
        ax.text(x0 + 0.6, j + .5, f'  {16 - j}', va='center')
        ax.text(x0 + 0.6 + 0.15, j + .5, f'  {jats_names[16 - j]}', va='center')
        ax.add_patch(r1)
        ax.axhline(j, color='k')

    j = len(types_colors)
    ax.text(
        0.05, j + 1 + 2, '''Bright colors - Lateral types (Alpha-Delta).
        Deep colors - Central types (Beta-Gamma).

        Types are in standard Talanov's order. 8 bright rainbow colors
        from red to pink are mapped to Alpha and Delta quadras.
        Deep/Bright switch would be Rational/Irrational switch to
        quasi-identity type.'''.replace(8 * ' ', ''),
        va='center')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, j + 2 + 4)
    ax.axis('off')

    display(HTML(f'''<img src="{mh.img(name='types_colors')}" width="300">'''))


# plot()
