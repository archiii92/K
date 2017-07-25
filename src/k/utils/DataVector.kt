package k.utils

import java.io.FileNotFoundException
import java.util.*
import kotlin.collections.ArrayList

fun readData(trainData: ArrayList<DataVector>, testData: ArrayList<DataVector>, trainTestDivide: Int, dataFileName: String, inputLayerSize: Int, outputLayerSize: Int){
    val fileIn = ClassLoader.getSystemResourceAsStream(dataFileName) ?: throw FileNotFoundException(dataFileName)

    val scanner = Scanner(fileIn)
    val readedLines = ArrayList<String>()
    while (scanner.hasNext()) {
        readedLines.add(scanner.nextLine())
    }
    scanner.close()

    val dataValues = ArrayList<Double>()

    for (i in readedLines.indices){
        if (i == 0){
            continue;
        }
        val dataString = readedLines[i].split(",")
        dataValues.add(dataString[1].toDouble())
    }

    val trainDataCount = (dataValues.count() - inputLayerSize - outputLayerSize) * trainTestDivide / 100;

    var i = 0
    while(i < dataValues.count() - inputLayerSize - outputLayerSize){
        val dataVector = DataVector(inputLayerSize, outputLayerSize)
        var j = 0

        while(j < inputLayerSize){
            dataVector.Window[j] = dataValues[i + j]
            j++
        }

        j = 0;
        while(j < outputLayerSize){
            dataVector.Forecast[j] = dataValues[i + inputLayerSize + j]
            j++
        }
        if (i < trainDataCount){
            trainData.add(dataVector)
        } else {
            testData.add(dataVector)
        }
        i++
    }
}

class DataVector constructor(windowSize: Int, forecastSize: Int){
    val Window: DoubleArray = kotlin.DoubleArray(windowSize)
    val Forecast: DoubleArray = kotlin.DoubleArray(forecastSize)
}