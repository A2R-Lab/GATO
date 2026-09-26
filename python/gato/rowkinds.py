"""Row-group KIND and MECHANISM ids — the Python mirror of ``gato/bsqp/rowgroups.cuh``
(``enum Kind`` / ``enum Mechanism``). ``BSQP.get_row_groups()`` reports these ids;
the constraint API (``interface.py``), the KKT certificate and the tests import
them from here instead of re-declaring literals."""

# rows::Kind
KIND_BOX_Q = 0        # g_i = q_i        (state block, ACTUATED rows from the URDF table)
KIND_BOX_QD = 1       # g_i = qd_i
KIND_BOX_U = 2        # g_i = u_i        (control block)
KIND_EE_POS = 3       # g_i = ee_pos_i(q_k), i < 3 (terminal EE row; cooperative FK)
KIND_LIN_U = 4        # g_i = C[i,:]·u_k + d_i (dense control map; cone=1 = second-order cone)
KIND_COLLISION = 5    # g_i = clearance of collision sphere i (one-sided, per knot; band state)
KIND_CONTACT_POS = 6  # g_i = p_f(q_k)[axis] - tgt_k[i], i = 3f + axis (per-knot contact-frame residual)

KIND_NAMES = {KIND_BOX_Q: "BOX_Q", KIND_BOX_QD: "BOX_QD", KIND_BOX_U: "BOX_U", KIND_EE_POS: "EE_POS",
              KIND_LIN_U: "LIN_U", KIND_COLLISION: "COLLISION", KIND_CONTACT_POS: "CONTACT_POS"}

# vector kinds are gated per KNOT by bit 0 of the row-activity mask (an SOC cone couples its rows;
# collision groups have more rows than the dense mask holds)
def is_vector_kind(kind, cone):
    return bool(cone) or int(kind) == KIND_COLLISION

# rows::Mechanism
MECH_TELEMETRY = 0
MECH_BARRIER_RELAXED = 1
MECH_ADMM = 2
MECH_AL = 3
MECHS = {"telemetry": MECH_TELEMETRY, "barrier": MECH_BARRIER_RELAXED, "admm": MECH_ADMM, "al": MECH_AL}

# BLOCK
BLOCK_X, BLOCK_U = 0, 1
