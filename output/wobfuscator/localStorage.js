
// === Wobfuscator-lite runtime ===
const __wobfBytes = Uint8Array.from(Buffer.from("AGFzbQEAAAABBgFgAX8BfwMCAQAFAwEAAQcWAgZtZW1vcnkCAAlnZXRPZmZzZXQAAAroAgHlAgAgAEEARgR/QQAFIABBAUYEf0EGBSAAQQJGBH9BFAUgAEEDRgR/QSIFIABBBEYEf0EuBSAAQQVGBH9BNwUgAEEGRgR/QT4FIABBB0YEf0HJAAUgAEEIRgR/QdEABSAAQQlGBH9B1wAFIABBCkYEf0HdAAUgAEELRgR/QekABSAAQQxGBH9B9AAFIABBDUYEf0H/AAUgAEEORgR/QYoBBSAAQQ9GBH9BmAEFIABBEEYEf0GlAQUgAEERRgR/QbABBSAAQRJGBH9BvQEFIABBE0YEf0HIAQUgAEEURgR/QdIBBSAAQRVGBH9B3QEFIABBFkYEf0HnAQUgAEEXRgR/QfMBBSAAQRhGBH9B/gEFIABBGUYEf0GJAgUgAEEaRgR/QZQCBSAAQRtGBH9BoAIFIABBHEYEf0GrAgUgAEEdRgR/QbYCBUEACwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwvvAx4AQQALBmphaW1lAABBBgsObm9tYnJlVXN1YXJpbwAAQRQLDm5vbWJyZVVzdWFyaW8AAEEiCwwjYnRuQm90b25lcwAAQS4LCSNib3RvbmVzAABBNwsHI2ZvbmRvAABBPgsLY29sb3JGb25kbwAAQckACwhiZy1kYXJrAABB0QALBmNsaWNrAABB1wALBmNsaWNrAABB3QALDGJ0bi1wcmltYXJ5AABB6QALC2JnLXByaW1hcnkAAEH0AAsLY29sb3JGb25kbwAAQf8ACwtiZy1wcmltYXJ5AABBigELDmJ0bi1zZWNvbmRhcnkAAEGYAQsNYmctc2Vjb25kYXJ5AABBpQELC2NvbG9yRm9uZG8AAEGwAQsNYmctc2Vjb25kYXJ5AABBvQELC2J0bi1kYW5nZXIAAEHIAQsKYmctZGFuZ2VyAABB0gELC2NvbG9yRm9uZG8AAEHdAQsKYmctZGFuZ2VyAABB5wELDGJ0bi1zdWNjZXNzAABB8wELC2JnLXN1Y2Nlc3MAAEH+AQsLY29sb3JGb25kbwAAQYkCCwtiZy1zdWNjZXNzAABBlAILDGJ0bi13YXJuaW5nAABBoAILC2JnLXdhcm5pbmcAAEGrAgsLY29sb3JGb25kbwAAQbYCCwtiZy13YXJuaW5nAA==", "base64"));
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

const nombre = __wstr__(0);
localStorage.setItem(__wstr__(1), nombre);
const nombreLocalStorage = localStorage.getItem(__wstr__(2));
console.log(nombreLocalStorage);
const btnBotones = document.querySelector(__wstr__(3));
const botones = document.querySelector(__wstr__(4));
const fondo = document.querySelector(__wstr__(5));
(() => {
    const colorBG = localStorage.getItem(__wstr__(6));
    console.log(colorBG);
    if (colorBG === null) {
        fondo.className = __wstr__(7);
    } else {
        fondo.className = colorBG;
    }
})();
(() => {
    btnBotones.addEventListener(__wstr__(8), agregarBotones);
    botones.addEventListener(__wstr__(9), delegacion);
})();
function agregarBotones(e) {
    e.preventDefault();
    botones.innerHTML = `
		<button class="btn btn-primary">primary</button>
		<button class="btn btn-secondary">secondary</button>
		<button class="btn btn-danger">danger</button>
		<button class="btn btn-success">success</button>
		<button class="btn btn-warning">warning</button>
	`;
}
function delegacion(e) {
    e.preventDefault();
    console.log(e.target.classList[1]);
    const colorBoton = e.target.classList[1];
    switch (colorBoton) {
    case __wstr__(10):
        fondo.className = __wstr__(11);
        localStorage.setItem(__wstr__(12), __wstr__(13));
        break;
    case __wstr__(14):
        fondo.className = __wstr__(15);
        localStorage.setItem(__wstr__(16), __wstr__(17));
        break;
    case __wstr__(18):
        fondo.className = __wstr__(19);
        localStorage.setItem(__wstr__(20), __wstr__(21));
        break;
    case __wstr__(22):
        fondo.className = __wstr__(23);
        localStorage.setItem(__wstr__(24), __wstr__(25));
        break;
    case __wstr__(26):
        fondo.className = __wstr__(27);
        localStorage.setItem(__wstr__(28), __wstr__(29));
        break;
    }
}
