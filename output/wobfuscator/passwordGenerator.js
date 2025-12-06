
// === Wobfuscator-lite runtime ===
const __wobfBytes = Uint8Array.from(Buffer.from("AGFzbQEAAAABBgFgAX8BfwMCAQAFAwEAAQcWAgZtZW1vcnkCAAlnZXRPZmZzZXQAAAq/AwG8AwAgAEEARgR/QQAFIABBAUYEf0EFBSAAQQJGBH9BGAUgAEEDRgR/QSwFIABBBEYEf0HgAAUgAEEFRgR/QZQBBSAAQQZGBH9ByAEFIABBB0YEf0HPAQUgAEEIRgR/QdsBBSAAQQlGBH9B4QEFIABBCkYEf0HvAQUgAEELRgR/QfUBBSAAQQxGBH9BggIFIABBDUYEf0GIAgUgAEEORgR/QZICBSAAQQ9GBH9BngIFIABBEEYEf0GkAgUgAEERRgR/Qa0CBSAAQRJGBH9BuwIFIABBE0YEf0HBAgUgAEEURgR/Qc0CBSAAQRVGBH9B0wIFIABBFkYEf0HYAgUgAEEXRgR/QeQCBSAAQRhGBH9B6gIFIABBGUYEf0HrAgUgAEEaRgR/QewCBSAAQRtGBH9B7gIFIABBHEYEf0HwAgUgAEEdRgR/Qf8CBSAAQR5GBH9BjgMFIABBH0YEf0GUAwUgAEEgRgR/QaMDBSAAQSFGBH9BqAMFIABBIkYEf0G4AwUgAEEjRgR/Qb8DBSAAQSRGBH9BzwMFQQALCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwuxBSUAQQALBSNhcHAAAEEFCxMjbnVtZXJvLWNhcmFjdGVyZXMAAEEYCxQwIDEgMiAzIDQgNSA2IDcgOCA5AABBLAs0fiAhIEAgIyAkICUgXiAmICogKCApIF8gKyAtID0gWyB7IH0gXSA7IDogLCA8IC4gPyAvAABB4AALNEEgQiBDIEQgRSBGIEcgSCBJIEogSyBMIE0gTiBPIFAgUSBSIFMgVCBVIFYgVyBYIFkgWgAAQZQBCzRhIGIgYyBkIGUgZiBnIGggaSBqIGsgbCBtIG4gbyBwIHEgciBzIHQgdSB2IHcgeCB5IHoAAEHIAQsHc3VibWl0AABBzwELDGJ0bi1tYXMtdW5vAABB2wELBmNsaWNrAABB4QELDmJ0bi1tZW5vcy11bm8AAEHvAQsGY2xpY2sAAEH1AQsNYnRuLXNpbWJvbG9zAABBggILBmNsaWNrAABBiAILCnNpbWJvbG9zIAAAQZICCwxidG4tbnVtZXJvcwAAQZ4CCwZjbGljawAAQaQCCwludW1lcm9zIAAAQa0CCw5idG4tbWF5dXNjdWxhAABBuwILBmNsaWNrAABBwQILDG1heXVzY3VsYXMgAABBzQILBmZhbHNlAABB0wILBW5vbmUAAEHYAgsMYnRuLWdlbmVyYXIAAEHkAgsGY2xpY2sAAEHqAgsBAABB6wILAQAAQewCCwIgAABB7gILAiAAAEHwAgsPaW5wdXQtcGFzc3dvcmQAAEH/AgsPaW5wdXQtcGFzc3dvcmQAAEGOAwsGY2xpY2sAAEGUAwsPaW5wdXQtcGFzc3dvcmQAAEGjAwsFY29weQAAQagDCxAuYWxlcnRhLWNvcGlhZG8AAEG4AwsHYWN0aXZlAABBvwMLEC5hbGVydGEtY29waWFkbwAAQc8DCwdhY3RpdmUA", "base64"));
const __wobfModule = new WebAssembly.Module(__wobfBytes);
const __wobfInstance = new WebAssembly.Instance(__wobfModule);
const __wobfMemory = __wobfInstance.exports.memory;
const __wobfGetOffset = __wobfInstance.exports.getOffset;

function __wstr__(idx) {
    const mem = new Uint8Array(__wobfMemory.buffer);
    let off = __wobfGetOffset(idx);
    const chars = [];
    while (mem[off] !== 0) {
        chars.push(mem[off++]);
    }
    return String.fromCharCode(...chars);
}
// === End Wobfuscator-lite runtime ===

(function () {
    var app = document.querySelector(__wstr__(0));
    var inputCaracteres = document.querySelector(__wstr__(1));
    var configuracion = {
        caracteres: parseInt(inputCaracteres.value),
        simbolos: true,
        numeros: true,
        mayusculas: true,
        minusculas: true
    };
    var caracteres = {
        numeros: __wstr__(2),
        simbolos: __wstr__(3),
        mayusculas: __wstr__(4),
        minusculas: __wstr__(5)
    };
    app.addEventListener(__wstr__(6), function (e) {
        e.preventDefault();
    });
    app.elements.namedItem(__wstr__(7)).addEventListener(__wstr__(8), () => {
        configuracion.caracteres++;
        inputCaracteres.value = configuracion.caracteres;
    });
    app.elements.namedItem(__wstr__(9)).addEventListener(__wstr__(10), () => {
        if (configuracion.caracteres > 1) {
            configuracion.caracteres--;
            inputCaracteres.value = configuracion.caracteres;
        }
    });
    app.elements.namedItem(__wstr__(11)).addEventListener(__wstr__(12), function () {
        btnToggle(this);
        configuracion.simbolos = !configuracion.simbolos;
        console.log(__wstr__(13) + configuracion.simbolos);
    });
    app.elements.namedItem(__wstr__(14)).addEventListener(__wstr__(15), function () {
        btnToggle(this);
        configuracion.numeros = !configuracion.numeros;
        console.log(__wstr__(16) + configuracion.numeros);
    });
    app.elements.namedItem(__wstr__(17)).addEventListener(__wstr__(18), function () {
        btnToggle(this);
        configuracion.mayusculas = !configuracion.mayusculas;
        console.log(__wstr__(19) + configuracion.mayusculas);
    });
    let btnToggle = elemento => {
        elemento.classList.toggle(__wstr__(20));
        elemento.childNodes[0].nextElementSibling.classList.toggle(__wstr__(21));
    };
    app.elements.namedItem(__wstr__(22)).addEventListener(__wstr__(23), function () {
        generatePassword();
    });
    let generatePassword = () => {
        let caracteresFinales = __wstr__(24);
        let password = __wstr__(25);
        for (propiedad in configuracion) {
            if (configuracion[propiedad] == true) {
                caracteresFinales += caracteres[propiedad] + __wstr__(26);
            }
        }
        caracteresFinales = caracteresFinales.trim();
        caracteresFinales = caracteresFinales.split(__wstr__(27));
        for (var i = 0; i < configuracion.caracteres; i++) {
            password += caracteresFinales[Math.floor(Math.random() * caracteresFinales.length)];
        }
        app.elements.namedItem(__wstr__(28)).value = password;
    };
    app.elements.namedItem(__wstr__(29)).addEventListener(__wstr__(30), () => {
        copyPassword();
    });
    let copyPassword = () => {
        app.elements.namedItem(__wstr__(31)).select();
        document.execCommand(__wstr__(32));
        document.querySelector(__wstr__(33)).classList.add(__wstr__(34));
        setTimeout(function () {
            document.querySelector(__wstr__(35)).classList.remove(__wstr__(36));
        }, 2000);
    };
    generatePassword();
}());
