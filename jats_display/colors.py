from matplotlib.colors import to_rgb

_types_colors = (
    'bright red', 'tangerine', 'bright yellow', 'apple green',
    'magenta', 'purple', 'blue', 'teal blue',
    'grass green', 'gold', 'dark orange', 'blood red',
    'aqua blue', 'azure', 'bright purple', 'bright pink',
)
types_color_names = (
    'bright red', 'bright orange', 'bright yellow', 'bright green',
    'dark magenta', 'dark purple', 'dark blue', 'dark aqua',
    'dark green', 'dark yellow', 'dark orange', 'dark red',
    'bright aqua', 'bright blue', 'bright purple', 'bright magenta',
)
# ярко-/темно- красный, оранжевый, жёлтый, зелёный, бирюзовый,
# синий, фиолетовый, пурпурный
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
        1: 'NeT-ENTP', 2: 'TiN-INTJ', 3: 'SiF-ISFP', 4: 'FeS-ESFJ',
        5: 'SeT-ESTP', 6: 'TiS-ISTJ', 7: 'NiF-INFP', 8: 'FeN-ENFJ',
        9: 'SeF-ESFP', 10: 'FiS-ISFJ', 11: 'NiT-INTP', 12: 'TeN-ENTJ',
        13: 'NeF-ENFP', 14: 'FiN-INFJ', 15: 'SiT-ISTP', 16: 'TeS-ESTJ'}

    fig = plt.figure(figsize=[4.3, 6])
    ax = fig.add_axes([0, 0, 1, 1])

    for j, (name, color) in enumerate(zip(reversed(types_color_names), reversed(types_colors))):
        r1 = mpatch.Rectangle((0, j), 1, 1, color=color)
        ax.text(1, j+.5, f'  type {16 - j}: {name} ({jats_names[16 - j]})', va='center')
        ax.add_patch(r1)
        ax.axhline(j, color='k')

    j = len(types_colors)
    ax.text(
        0., j + 1, '''Bright colors - pacific types,
        Dark colors - resolute types,
        Dark/Light switch - J/P switch.'''.replace(8 * ' ', ''),
        va='center')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, j + 2)
    ax.axis('off')

    display(HTML(f'''<img src="{mh.img(plt, name='types_colors')}" width="300">'''))


# plot()
