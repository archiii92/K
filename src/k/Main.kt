package k

import k.nets.MLP

fun main(args: Array<String>) {
    val mlp = MLP(3, 12, 1, "gold.txt", 90)
    mlp.prepareData()
    mlp.buildNetwork()
    mlp.learn()
}