package k.neurons

///* Нейрон с функцией активации в виде гиперболического тангенса */
//class HyperbolicTangentNeuron : Neuron {
//    constructor(inputSize: Int, outputSize: Int) {
//        val inputValues: ArrayList<Double> = ArrayList(inputSize)
//        val weights: ArrayList<Double> = ArrayList(inputSize)
//    }
//
//    /* Гиперболический тангенс */
//    fun activationFunction(x: Double): Double {
//        val a = Math.pow(Math.E, x)
//        val b = Math.pow(Math.E, -x)
//        return (a - b) / (a + b)
//    }
//
//    /* Производная гиперболического тангенса */
//    fun activationFunctionDerivative(x: Double): Double {
//        val v = activationFunction(x)
//        return v * (1 - v)
//    }
//}