"""
node {
  name: "Custom_1"
  op: "FusedCwise"
  input: "f_product_cart_all_1d"
  input: "f_product_cart_order_center_1d"
  attr {
    key: "Targs"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
  attr {
    key: "Toutputs"
    value {
      list {
        type: DT_INT32
        type: DT_INT32
      }
    }
  }
}
"""

dtype_key = "Toutputs"
print(node.attr[dtype_key].list.type[0])