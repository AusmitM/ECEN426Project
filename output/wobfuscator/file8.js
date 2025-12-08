
// === Wobfuscator-lite runtime ===
const __wobfBytes = Uint8Array.from(Buffer.from("AGFzbQEAAAABBgFgAX8BfwMCAQAFAwEAAQcWAgZtZW1vcnkCAAlnZXRPZmZzZXQAAApeAVwAIABBAEYEf0EABSAAQQFGBH9BCQUgAEECRgR/QRMFIABBA0YEf0EcBSAAQQRGBH9BIwUgAEEFRgR/QSQFIABBBkYEf0EqBSAAQQdGBH9BNQVBAAsLCwsLCwsLCwtfCABBAAsJdG9kb0Zvcm0AAEEJCwp0b2RvSW5wdXQAAEETCwlpdGVtTGlzdAAAQRwLB3N1Ym1pdAAAQSMLAQAAQSQLBmVtcHR5AABBKgsLYWZ0ZXJiZWdpbgAAQTULAQA=", "base64"));
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

let todoForm = document.getElementById(__wstr__(0));
let todoInput = document.getElementById(__wstr__(1));
let itemList = document.getElementById(__wstr__(2));
todoForm.addEventListener(__wstr__(3), function (e) {
    e.preventDefault();
    if (todoInput.value == __wstr__(4)) {
        inputoEmpty();
        return false;
    }
    addItem(todoInput.value);
});
function inputoEmpty() {
    console.log(__wstr__(5));
}
function addItem(item) {
    let listItem = `<li>${ item } <button onclick="removeItem(this)">x</button>`;
    list.insertAdjacentHTML(__wstr__(6), listItem);
    todoInput.value = __wstr__(7);
    todoInput.focus();
}
function removeItem(itemToDelete) {
    itemToDelete.parentElement.remove();
}
