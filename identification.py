from adjustment_formulae import *
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


def get_default_nd_to_size(dot_file, hidden_nd_names):
    nodes, _ = DotTool.read_dot_file(dot_file)
    nd_to_size = {}
    for nd in nodes:
        nd_to_size[nd] = 2
    for nd_name in hidden_nd_names:
        assert nd_name in nodes, (f"hidden node '{nd_name}'"
                                  f" not in full node list")
        nd_to_size[nd_name] = 3
    return nd_to_size


def calculate_ampu_and_full_pots(bnet):
    node_list = list(bnet.nodes)

    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    ampu_pot = None
    for nd in node_list:
        if nd != nd_x:
            if not ampu_pot:
                ampu_pot = nd.potential
            else:
                ampu_pot = ampu_pot * nd.potential
    full_pot = ampu_pot * nd_x.potential
    return ampu_pot, full_pot


def calculate_do_query_probs(bnet, ampu_pot, full_pot):
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')
    ampu_prob_y_bar_x = None
    full_prob_y_bar_x = None
    for final_pot in [ampu_pot, full_pot]:
        arr_xy = final_pot.get_new_marginal([nd_x, nd_y]).pot_arr
        pot_y_bar_x = DiscreteCondPot(
            False, [nd_x, nd_y], arr_xy)
        pot_y_bar_x.normalize_self()
        if final_pot == ampu_pot:
            ampu_prob_y_bar_x = pot_y_bar_x.pot_arr
        elif final_pot == full_pot:
            full_prob_y_bar_x = pot_y_bar_x.pot_arr
        else:
            assert None

    return ampu_prob_y_bar_x, full_prob_y_bar_x


def compare_two_do_queries(dot_file,
                           hidden_nd_names,
                           nd_to_size=None,
                           verbose=True):
    if not nd_to_size:
        nd_to_size = get_default_nd_to_size(dot_file, hidden_nd_names)
    nodes, arrows = DotTool.read_dot_file(dot_file)

    bnet = create_random_bnet(
        nodes,
        arrows,
        nd_to_size)

    for bnet_str in ["bnet1", "bnet2"]:
        if bnet_str == "bnet2":
            print("------------------------------")
            randomize_these_nodes(bnet, hidden_nd_names)
        print(f"Random {bnet_str}:")
        if verbose:
            print(bnet)
        ampu_pot, full_pot = calculate_ampu_and_full_pots(bnet)
        ampu_prob_y_bar_x, full_prob_y_bar_x = \
            calculate_do_query_probs(bnet, ampu_pot, full_pot)
        print()
        print(f"full P(y|x) for {bnet_str}:")
        pprint(full_prob_y_bar_x)
        print()
        print(f"amputated P(y|x) for {bnet_str}: (REQUIRES RCT)")
        pprint(ampu_prob_y_bar_x)
        dotf_str_to_adj_method = {
            "back-door": get_backdoor_adjustment_prob,
            "front-door": get_frontdoor_adjustment_prob,
            "napkin": get_napkin_adjustment_prob}
        adj_method = None
        for dotf_str in dotf_str_to_adj_method:
            if dotf_str in dot_file:
                adj_method = dotf_str_to_adj_method[dotf_str]
                break
        if adj_method:
            print()
            print(f"adjusted P(y|x) from full_pot for {bnet_str}:")
            pprint(adj_method(bnet, full_pot))
            print()
            print(f"adjusted P(y|x) from ampu_pot for {bnet_str}:"
                  f"(REQUIRES RCT)")
            pprint(adj_method(bnet, ampu_pot))


if __name__ == "__main__":
    def main_backdoor(draw, verbose):
        compare_two_do_queries(dot_file="dot_atlas/back-door.dot",
                               hidden_nd_names=[],
                               verbose=verbose)


    def main_frontdoor(draw, verbose):
        compare_two_do_queries(dot_file="dot_atlas/front-door.dot",
                               hidden_nd_names=["h"],
                               verbose=verbose)


    def main_napkin(draw, verbose):
        compare_two_do_queries(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               verbose=verbose)


    # main_backdoor(False, False)
    # main_frontdoor(False, False)
    main_napkin(False, False)
