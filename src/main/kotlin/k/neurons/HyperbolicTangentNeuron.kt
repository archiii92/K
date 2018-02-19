//package k.neurons
//
///* Нейрон с функцией активации в виде гиперболического тангенса */
//class HyperbolicTangentNeuron(inputVectorSize: Int) : AbstractMLPNeuron(inputVectorSize) {
//
//    var k: Double = 1.0 // Коэффициент гиперболического тангенса
//
//    /* Гиперболический тангенс */
//    override fun activationFunction(x: Double): Double {
//        val a = Math.exp(k * x)
//        val b = Math.exp(-k * x)
//        return (a - b) / (a + b)
//    }
//
//    /* Производная гиперболического тангенса */
//    override fun activationFunctionDerivative(x: Double): Double {
//        return 1 - Math.pow(activationFunction(x), 2.0)
//    }
//}