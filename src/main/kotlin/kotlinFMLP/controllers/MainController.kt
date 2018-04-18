package kotlinFMLP.controllers

import org.springframework.web.bind.annotation.CrossOrigin
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController

//class SimulatedAnnealingSettings() {
//    lateinit var initialTemperature: Number
//    lateinit var warmingKeepPercent: Number
//
//    constructor(
//            initialTemperature: Number,
//            warmingKeepPercent: Number
//    ): this() {
//        this.initialTemperature = initialTemperature
//        this.warmingKeepPercent = warmingKeepPercent
//    }
//}

class ForecastSettings {
    lateinit var fileName: String
    lateinit var trainTestDivide: Number
    lateinit var fuzzyLayerSize: Number
    lateinit var hiddenLayerSize: Number
    lateinit var inputLayerSize: Number
    lateinit var selectedNetwork: String
    lateinit var selectedAlgorithm: String

    lateinit var initialTemperature: Number
    lateinit var warmingKeepPercent: Number

    lateinit var particleCount: Number
    lateinit var iterationCount: Number
    lateinit var k: Number
    lateinit var φg: Number
    lateinit var φp: Number

//    lateinit var algorithmParameters: SimulatedAnnealingSettings
//    lateinit var initialTemperature: Number
//    lateinit var warmingKeepPercent: Number

//    constructor(
//            fileName: String,
//            testTrainDivide: Number,
//            fuzzyLayerSize: Number,
//            hiddenLayerSize: Number,
//            inputLayerSize: Number,
//            selectedNetwork: String,
//            selectedAlgorithm: String
////            initialTemperature: Number,
////            warmingKeepPercent: Number
//    ): this() {
//        this.fileName = fileName
//        this.testTrainDivide = testTrainDivide
//        this.fuzzyLayerSize = fuzzyLayerSize
//        this.hiddenLayerSize = hiddenLayerSize
//        this.inputLayerSize = inputLayerSize
//        this.selectedNetwork = selectedNetwork
//        this.selectedAlgorithm = selectedAlgorithm
////        this.initialTemperature = initialTemperature
////        this.warmingKeepPercent = warmingKeepPercent
//    }
}

class ForecastResult {
    lateinit var initError: Number
    lateinit var afterFuzzyLayerInitError: Number
    lateinit var afterOptimizationError: Number
    lateinit var finalError: Number

    lateinit var realValues: DoubleArray
    lateinit var forecastValues: DoubleArray
    lateinit var forecastDates: Array<String>
}

@RestController
class MainController {

    @PostMapping("/forecast")
    @CrossOrigin(origins = ["http://localhost:8081"])
    fun forecast(@RequestBody forecastSettings: ForecastSettings): ForecastResult {

        val model = MainModel()

        return model.makeForecast(forecastSettings)
    }
}