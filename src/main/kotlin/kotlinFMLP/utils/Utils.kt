package kotlinFMLP.utils

import java.io.FileNotFoundException
import java.util.*
import kotlin.collections.ArrayList

var dif = 0.0
var min = 0.0
val data = ArrayList<DataVector>()
var trainDataCount = 0

fun readData(trainData: ArrayList<DataVector>, testData: ArrayList<DataVector>, trainTestDivide: Int, dataFileName: String, inputLayerSize: Int, outputLayerSize: Int) {
    val fileIn = ClassLoader.getSystemResourceAsStream("datasets/$dataFileName")
            ?: throw FileNotFoundException(dataFileName)

    val scanner = Scanner(fileIn)
    val readedLines = ArrayList<String>()
    while (scanner.hasNext()) {
        readedLines.add(scanner.nextLine())
    }
    scanner.close()

    val dataValues = DoubleArray(readedLines.size - 1)
    val dates = Array<String>(readedLines.size - 1) { "" }

    var max = Double.MIN_VALUE
    min = Double.MAX_VALUE

    for (i in readedLines.indices) {
        if (i == 0) {
            continue
        }
        val dataString = readedLines[i].split(",")

        val dataDate = dataString[0]
        val dataValue = dataString[1].toDouble()

        if (dataValue > max)
            max = dataValue
        else if (dataValue < min)
            min = dataValue

        dataValues[i - 1] = dataValue
        dates[i - 1] = dataDate
    }

    dif = max - min

    var i = 0
    while (i < dataValues.count() - inputLayerSize - outputLayerSize) {
        val dataVector = DataVector(inputLayerSize, outputLayerSize, dates[i])
        var j = 0

        while (j < inputLayerSize) {
            dataVector.Window[j] = dataValues[i + j]
            j++
        }

        j = 0
        while (j < outputLayerSize) {
            dataVector.Forecast[j] = dataValues[i + inputLayerSize + j]
            j++
        }

        data.add(dataVector)
        i++
    }

    trainDataCount = (dataValues.count() - inputLayerSize - outputLayerSize) * trainTestDivide / 100
    shuffleData(trainData, testData)
}

fun shuffleData(trainData: ArrayList<DataVector>, testData: ArrayList<DataVector>){
    Collections.shuffle(data)
    for (j in data.indices) {
        if (j < trainDataCount) {
            trainData.add(data[j])
        } else {
            testData.add(data[j])
        }
    }
}

fun normalize(value: Double): Double {
    return (value - min) / dif
}

fun denormalize(value: Double): Double {
    return value * dif + min
}

fun Double.format(digits: Int = 6) = java.lang.String.format("%.${digits}f", this)!!

fun getEuclideanDistance(v1: DoubleArray, v2: DoubleArray): Double {
    return Math.sqrt(v2.indices.sumByDouble { Math.pow(v1[it] - v2[it], 2.0) })
}

fun DoubleArray.toFormatString(digits: Int): String {
    val iMax = this.size - 1
    if (iMax == -1)
        return "[]"

    val b = StringBuilder()
    b.append('[')
    var i = 0
    while (true) {
        b.append(this[i].format(digits))
        if (i == iMax)
            return b.append(']').toString()
        b.append(", ")
        i++
    }
}