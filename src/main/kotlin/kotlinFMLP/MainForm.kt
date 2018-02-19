package kotlinFMLP

import tornadofx.View
import tornadofx.hbox
import tornadofx.label

class MainForm : View("Main Form") {
    override val root = hbox {
        label("Hello world")
    }
}