package k

import k.nets.MLP

fun main(args: Array<String>) {
    val mlp = MLP()
    mlp.prepareData()
    mlp.buildNetwork()
    mlp.learn()
    mlp.test()
}