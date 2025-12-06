
// === Wobfuscator-lite runtime ===
const __wobfBytes = Uint8Array.from(Buffer.from("AGFzbQEAAAABBgFgAX8BfwMCAQAFAwEAAQcWAgZtZW1vcnkCAAlnZXRPZmZzZXQAAAqfAQGcAQAgAEEARgR/QQAFIABBAUYEf0ETBSAAQQJGBH9BIAUgAEEDRgR/QTQFIABBBEYEf0HGAAUgAEEFRgR/Qd8ABSAAQQZGBH9BhAEFIABBB0YEf0GKAQUgAEEIRgR/QZABBSAAQQlGBH9BkgEFIABBCkYEf0GUAQUgAEELRgR/QZYBBSAAQQxGBH9BnAEFQQALCwsLCwsLCwsLCwsLCwvpAQ0AQQALEy5sb2NhdGlvbi10aW1lem9uZQAAQRMLDS50ZW1wZXJhdHVyZQAAQSALFC50ZW1wZXJhdHVyZS1kZWdyZWUAAEE0CxIudGVtcGVyYXR1cmUgc3BhbgAAQcYACxkudGVtcGVyYXR1cmUtZGVzY3JpcHRpb24AAEHfAAslaHR0cHM6Ly9jb3JzLWFueXdoZXJlLmhlcm9rdWFwcC5jb20vAABBhAELBi5pY29uAABBigELBmNsaWNrAABBkAELAkYAAEGSAQsCQwAAQZQBCwJGAABBlgELBndoaXRlAABBnAELAl8A", "base64"));
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
    let long;
    let lat;
    let locationTimezone = document.querySelector(__wstr__(0));
    let temperatureSection = document.querySelector(__wstr__(1));
    let temperatureDegree = document.querySelector(__wstr__(2));
    let temperatureSpan = document.querySelector(__wstr__(3));
    let temperatureDescription = document.querySelector(__wstr__(4));
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(position => {
            long = position.coords.longitude;
            lat = position.coords.latitude;
            const proxy = __wstr__(5);
            const api = `${ proxy }https://api.darksky.net/forecast/3ed2820bdef835d0923968060af681dd/${ lat },${ long }`;
            fetch(api).then(response => {
                return response.json();
            }).then(data => {
                console.log(data);
                const {temperature, summary, icon} = data.currently;
                locationTimezone.textContent = data.timezone;
                temperatureDegree.textContent = temperature;
                temperatureDescription.textContent = summary;
                setIcons(icon, document.querySelector(__wstr__(6)));
                let celsius = (temperature - 32) * (5 / 9);
                temperatureSection.addEventListener(__wstr__(7), () => {
                    if (temperatureSpan.textContent === __wstr__(8)) {
                        temperatureSpan.textContent = __wstr__(9);
                        temperatureDegree.textContent = Math.floor(celsius);
                    } else {
                        temperatureSpan.textContent = __wstr__(10);
                        temperatureDegree.textContent = temperature;
                    }
                });
            });
        });
    }
    function setIcons(icon, iconID) {
        const skycons = new Skycons({ color: __wstr__(11) });
        const currentIcon = icon.replace(/-/g, __wstr__(12)).toUpperCase();
        skycons.play();
        return skycons.set(iconID, Skycons[currentIcon]);
    }
}());
