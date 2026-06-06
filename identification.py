from potentials.DiscreteCondPot import *
from graphs.BayesNet import *
from pprint import pprint


def create_random_bnet(nodes,
                       arrows,
                       nd_to_size):
    """
    This method returns a BayesNet object whose structure is given by
    'nodes' and 'arrows'. The TPM (transition probability matrix) for
    each node is created at random, with the only constraint being that
    the number of states of each node be as specified by the input
    'nd_to_size'.

    Parameters
    ----------
    nodes: list[str]
        example: ['a', 'b', 'c']
    arrows: list[tuple[str, str]]
        example: [('a', 'b'), ('a', 'c')]
    nd_to_size: dict[str, int]
        dictionary mapping node name to its size (i.e., the number of
        values or states)

    Returns
    -------
    BayesNet

    """
    bnet_nodes = []
    found_x = False
    found_y = False
    for name in nodes:
        if name == "x":
            found_x = True
        if name == "y":
            found_y = True
    assert found_x and found_y, \
        "can't find 'x' or 'y' in list of node names"

    for k, node_name in enumerate(nodes):
        nd = BayesNode(k, name=node_name)
        nd.size = nd_to_size[node_name]
        bnet_nodes.append(nd)
    bnet = BayesNet(set(bnet_nodes))
    for arrow in arrows:
        pa_nd = bnet.get_node_named(arrow[0])
        child_nd = bnet.get_node_named(arrow[1])
        child_nd.add_parent(pa_nd)

    # print("ccvv", bnet.nodes)
    for nd in bnet_nodes:
        ord_nodes = list(nd.parents) + [nd]
        # print("llkjh", ord_nodes)
        nd.potential = DiscreteCondPot(False, ord_nodes)
        nd.potential.set_to_random()
        nd.potential.normalize_self()
    # print("aadf", bnet)
    return bnet


def randomize_these_nodes(bnet, some_node_names):
    for name in some_node_names:
        nd = bnet.get_node_named(name)
        nd.potential.set_to_random()
        nd.potential.normalize_self()


def calculate_do_query_prob(bnet):
    node_list = list(bnet.nodes)
    amp_pot = None
    for nd in node_list:
        if nd.name != "x":
            amp_pot = nd.potential
            break
    for nd in node_list:
        if nd.name != "x":
            amp_pot = amp_pot * nd.potential

    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    full_pot = amp_pot * nd_x.potential

    amp_prob_y_bar_x = None
    full_prob_y_bar_x = None
    for final_pot in [amp_pot, full_pot]:
        arr_y_bar_x = final_pot.get_new_marginal([nd_x, nd_y]).pot_arr
        pot_y_bar_x = DiscreteCondPot(
            False, [nd_x, nd_y], arr_y_bar_x)
        pot_y_bar_x.normalize_self()
        if final_pot == amp_pot:
            amp_prob_y_bar_x = pot_y_bar_x.pot_arr
        elif final_pot == full_pot:
            full_prob_y_bar_x = pot_y_bar_x.pot_arr
        else:
            assert None

    return amp_prob_y_bar_x, full_prob_y_bar_x


def compare_two_do_queries(dot_file,
                           hidden_nd_names,
                           nd_to_size,
                           draw=True,
                           jupyter=False,
                           verbose=True):
    nodes, arrows = DotTool.read_dot_file(dot_file)

    bnet = create_random_bnet(
        nodes,
        arrows,
        nd_to_size)
    if draw:
        bnet.gv_draw(jupyter)

    print("Random bnet1:")
    if verbose:
        print(bnet)
    amp_prob_y_bar_x, full_prob_y_bar_x = calculate_do_query_prob(bnet)
    print()
    print("full P(y|x) for bnet1:")
    pprint(full_prob_y_bar_x)
    print()
    print("amputated P(y|x) for bnet1:")
    pprint(amp_prob_y_bar_x)

    randomize_these_nodes(bnet, hidden_nd_names)
    print("------------------------------")
    print("Random bnet2:")
    if verbose:
        print(bnet)
    amp_prob_y_bar_x, full_prob_y_bar_x = calculate_do_query_prob(bnet)
    print()
    print("full P(y|x) for bnet2:")
    pprint(full_prob_y_bar_x)
    print()
    print("amputated P(y|x) for bnet2:")
    pprint(amp_prob_y_bar_x)


if __name__ == "__main__":
    def main(draw, verbose):
        dot_file = "dot_atlas/napkin.dot"
        hidden_nd_names = ["u_1", "u_2"]

        nodes, _ = DotTool.read_dot_file(dot_file)
        nd_to_size = {}
        for nd in nodes:
            nd_to_size[nd] = 2
        for nd_name in hidden_nd_names:
            assert nd_name in nodes, (f"hidden node '{nd_name}'"
                                 f" not in full node list")
            nd_to_size[nd_name] = 3

        compare_two_do_queries(dot_file,
                               hidden_nd_names,
                               nd_to_size,
                               draw=draw,
                               jupyter=False,
                               verbose=verbose)
    main(draw=True, verbose=True)